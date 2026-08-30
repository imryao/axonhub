package modelhub

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

const (
	// DefaultBaseURL is the native ModelHub online API endpoint.
	DefaultBaseURL = "https://aidp.bytedance.net/api/modelhub/online"

	// DefaultEndpointPath is the Responses endpoint exposed by ModelHub.
	DefaultEndpointPath = "/responses"

	// APIKeyQueryParameter is the query parameter ModelHub uses for access keys.
	APIKeyQueryParameter = "ak"

	// DefaultMaxOutputTokens matches the limit used by modelhub-bridge when the
	// caller does not provide an explicit output-token limit.
	DefaultMaxOutputTokens int64 = 128000

	modelHubLogIDField         = "X-TT-logid"
	modelHubExtraField         = "extra"
	modelHubSessionMetadataKey = "modelhub_session_id"
)

var defaultModels = []string{
	"gpt-5.6-sol",
	"gpt-5.6-terra",
	"gpt-5.6-luna",
	"gpt-5.5-2026-04-24",
	"gpt-5.4-2026-03-05",
	"gpt-5.2-2025-12-11",
}

var modelHubAKInText = regexp.MustCompile(`(?i)(^|[?&])ak=[^&\s"'<>]+`)

// DefaultModels returns the model IDs commonly available through ModelHub.
// A copy is returned so callers cannot mutate the package defaults.
func DefaultModels() []string {
	return append([]string(nil), defaultModels...)
}

// Config holds native ModelHub outbound configuration.
type Config struct {
	// BaseURL is the ModelHub online API base, without the /responses suffix.
	BaseURL string `json:"base_url,omitempty"`

	// EndpointPath overrides the native Responses path. It is primarily useful
	// for channel endpoint overrides; the default is /responses.
	EndpointPath string `json:"endpoint_path,omitempty"`

	// APIKeyProvider supplies the ModelHub access key. The key is sent in the
	// request query as `ak`, never embedded in BaseURL.
	APIKeyProvider auth.APIKeyProvider `json:"-"`
}

// OutboundTransformer adapts AxonHub's OpenAI Responses payload to ModelHub's
// native HTTP contract. Responses conversion and stream parsing are delegated
// to the existing OpenAI Responses transformer.
type OutboundTransformer struct {
	*responses.OutboundTransformer

	apiKeyProvider auth.APIKeyProvider
	baseURL        string
	endpointPath   string
}

var (
	_ transformer.Outbound              = (*OutboundTransformer)(nil)
	_ transformer.PassThroughBodyPolicy = (*OutboundTransformer)(nil)
)

// NewOutboundTransformer creates a native ModelHub transformer from a static AK.
func NewOutboundTransformer(baseURL, apiKey string) (transformer.Outbound, error) {
	return NewOutboundTransformerWithConfig(&Config{
		BaseURL:        baseURL,
		APIKeyProvider: auth.NewStaticKeyProvider(apiKey),
	})
}

// NewOutboundTransformerWithConfig creates a native ModelHub transformer.
func NewOutboundTransformerWithConfig(config *Config) (*OutboundTransformer, error) {
	if config == nil {
		return nil, fmt.Errorf("config is nil")
	}

	if config.APIKeyProvider == nil {
		return nil, fmt.Errorf("API key provider is required")
	}

	baseURL := strings.TrimRight(strings.TrimSpace(config.BaseURL), "/")
	baseURL, err := normalizeAndValidateBaseURL(baseURL)
	if err != nil {
		return nil, err
	}

	configuredEndpointPath := strings.TrimSpace(config.EndpointPath)
	endpointPath := configuredEndpointPath
	if endpointPath == "" {
		endpointPath = DefaultEndpointPath
	}
	if !strings.HasPrefix(endpointPath, "/") || strings.ContainsAny(endpointPath, "?#") || strings.HasPrefix(endpointPath, "//") || strings.Contains(endpointPath, "..") {
		return nil, fmt.Errorf("ModelHub endpoint path must be an absolute path without query or fragment")
	}

	responsesOutbound, err := responses.NewOutboundTransformerWithConfig(&responses.Config{
		BaseURL:        baseURL,
		EndpointPath:   endpointPath,
		APIKeyProvider: config.APIKeyProvider,
	})
	if err != nil {
		return nil, fmt.Errorf("invalid ModelHub Responses transformer configuration: %w", err)
	}

	return &OutboundTransformer{
		OutboundTransformer: responsesOutbound,
		apiKeyProvider:      config.APIKeyProvider,
		baseURL:             baseURL,
		endpointPath:        configuredEndpointPath,
	}, nil
}

// ValidateBaseURL validates a ModelHub base URL without constructing a
// transformer. It is used by channel persistence validation so malformed URLs
// fail at configuration time rather than only when the channel cache reloads.
func ValidateBaseURL(baseURL string) error {
	_, err := normalizeAndValidateBaseURL(baseURL)
	return err
}

func normalizeAndValidateBaseURL(rawURL string) (string, error) {
	baseURL := strings.TrimRight(strings.TrimSpace(rawURL), "/")
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}

	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("invalid ModelHub base URL")
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return "", fmt.Errorf("ModelHub base URL must use http or https")
	}
	// Keeping credentials out of the URL prevents accidental persistence in
	// channel metadata, request errors, and proxy logs.
	if parsed.User != nil || parsed.RawQuery != "" || parsed.ForceQuery || parsed.Fragment != "" || parsed.RawFragment != "" || strings.Contains(baseURL, "#") || parsed.Opaque != "" {
		return "", fmt.Errorf("ModelHub base URL must not contain userinfo, query, or fragment")
	}

	return baseURL, nil
}

// TransformRequest adds ModelHub's query authentication and tracing envelope
// around the standard OpenAI Responses request produced by the delegated
// transformer.
func (t *OutboundTransformer) TransformRequest(ctx context.Context, llmReq *llm.Request) (*httpclient.Request, error) {
	if t == nil || t.OutboundTransformer == nil {
		return nil, fmt.Errorf("ModelHub transformer is nil")
	}
	if t.apiKeyProvider == nil {
		return nil, fmt.Errorf("ModelHub API key provider is nil")
	}

	apiKey := strings.TrimSpace(t.apiKeyProvider.Get(ctx))
	if apiKey == "" {
		return nil, fmt.Errorf("%w: ModelHub API key is required", transformer.ErrInvalidRequest)
	}

	httpReq, err := t.OutboundTransformer.TransformRequest(ctx, llmReq)
	if err != nil {
		return nil, err
	}

	logID := validCorrelationID(httpReq.RequestID)
	if logID == "" && llmReq != nil && llmReq.RawRequest != nil {
		logID = validCorrelationID(headerValueFold(llmReq.RawRequest.Headers, "X-TT-LOGID"))
	}
	if logID == "" {
		logID = uuid.NewString()
	}
	httpReq.RequestID = logID

	sessionID := resolveSessionID(ctx, llmReq)

	extraBytes, err := json.Marshal(map[string]string{"session_id": sessionID})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ModelHub extra metadata: %w", err)
	}
	extra := string(extraBytes)

	if httpReq.Headers == nil {
		httpReq.Headers = make(http.Header)
	}
	httpReq.Headers.Set("Content-Type", "application/json")
	httpReq.Headers.Set("Accept", "application/json")
	httpReq.Headers.Set("X-TT-LOGID", logID)
	httpReq.Headers.Set("extra", extra)
	// The delegated transformer uses bearer auth for ordinary OpenAI providers;
	// ModelHub authenticates with the query parameter below instead.
	httpReq.Headers.Del("Authorization")
	httpReq.Auth = nil

	httpReq.Query = make(url.Values)
	httpReq.Query.Set(APIKeyQueryParameter, apiKey)
	httpReq.SensitiveQueryParameters = []string{APIKeyQueryParameter}
	// Do not allow caller-supplied query parameters to override or leak into the
	// ModelHub request.
	httpReq.SkipInboundQueryMerge = true

	bodyForModelHub := httpReq.Body
	// The compact inbound transformer intentionally models only the documented
	// fields. Reuse the original JSON object when available so native ModelHub
	// controls added by newer clients are not discarded before forwarding.
	if llmReq != nil && llmReq.RequestType == llm.RequestTypeCompact && llmReq.RawRequest != nil && len(llmReq.RawRequest.Body) > 0 {
		bodyForModelHub = llmReq.RawRequest.Body
	}
	httpReq.Body, err = prepareModelHubBody(bodyForModelHub, llmReq, logID, extra)
	if err != nil {
		return nil, err
	}
	httpReq.JSONBody = append([]byte(nil), httpReq.Body...)

	// The Responses transformer has a fixed compact path. Honour a channel
	// endpoint override when one was explicitly configured, while retaining
	// /responses/compact as the native default for compact requests.
	if llmReq != nil && llmReq.RequestType == llm.RequestTypeCompact && t.endpointPath != "" {
		httpReq.URL = t.baseURL + t.endpointPath
	}

	return httpReq, nil
}

// AllowPassThroughBody deliberately disables raw body pass-through for this
// channel. The ModelHub envelope and query authentication must be installed on
// every attempt, even when the global/channel pass-through setting is enabled.
func (t *OutboundTransformer) AllowPassThroughBody(_ context.Context, _ *llm.Request, _ *httpclient.Request) bool {
	return false
}

// TransformError keeps ModelHub's AK out of diagnostic URLs while preserving
// the normal OpenAI Responses error shape. The HTTP client already redacts
// URLs in debug logs; this copy also covers persisted/request-level errors.
func (t *OutboundTransformer) TransformError(ctx context.Context, rawErr *httpclient.Error) *llm.ResponseError {
	if t == nil || t.OutboundTransformer == nil {
		return &llm.ResponseError{
			StatusCode: http.StatusInternalServerError,
			Detail: llm.ErrorDetail{
				Message: "ModelHub transformer is not initialized",
				Type:    "api_error",
			},
		}
	}
	if rawErr == nil {
		return t.OutboundTransformer.TransformError(ctx, nil)
	}

	cloned := *rawErr
	cloned.URL = redactModelHubURL(rawErr.URL)

	return t.OutboundTransformer.TransformError(ctx, &cloned)
}

func redactModelHubURL(rawURL string) string {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return modelHubAKInText.ReplaceAllString(rawURL, `${1}ak=<redacted>`)
	}

	query := parsed.Query()
	redacted := false
	for key := range query {
		if !strings.EqualFold(key, APIKeyQueryParameter) {
			continue
		}
		query[key] = []string{"<redacted>"}
		redacted = true
	}
	if redacted {
		parsed.RawQuery = query.Encode()
		return parsed.String()
	}

	return modelHubAKInText.ReplaceAllString(parsed.String(), `${1}ak=<redacted>`)
}

func prepareModelHubBody(body []byte, llmReq *llm.Request, logID, extra string) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(bytes.TrimSpace(body), &payload); err != nil {
		return nil, fmt.Errorf("failed to decode ModelHub request body: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("failed to decode ModelHub request body: expected a JSON object")
	}
	if llmReq != nil && llmReq.Model != "" {
		modelJSON, err := json.Marshal(llmReq.Model)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal ModelHub model: %w", err)
		}
		deleteJSONKeysFold(payload, "model")
		payload["model"] = modelJSON
	}

	isCompact := llmReq != nil && llmReq.RequestType == llm.RequestTypeCompact
	if !isCompact {
		normalizeModelHubInput(payload)
		// The native regular Responses endpoint accepts the subset assembled by
		// modelhub-bridge. Remove client/proxy-only controls that it does not
		// understand, while retaining response content, tools, sampling, and
		// reasoning fields.
		deleteJSONKeysFold(payload,
			"client_metadata",
			"include",
			"metadata",
			"parallel_tool_calls",
			"previous_response_id",
			"prompt_cache_key",
			"response_format",
			"service_tier",
			"store",
			"stream_options",
			"user",
			"max_tool_calls",
			"prompt_cache_retention",
			"truncation",
		)

		stream := false
		if llmReq != nil && llmReq.Stream != nil {
			stream = *llmReq.Stream
		}
		deleteJSONKeysFold(payload, "stream")
		payload["stream"] = json.RawMessage(strconv.FormatBool(stream))
	}

	normalizeModelHubMaxOutputTokens(payload)
	normalizeModelHubReasoning(payload)
	deleteJSONKeysFold(payload, "session_id", "clientRequestBody", modelHubLogIDField, modelHubExtraField)
	removeNestedSessionID(payload, "metadata")
	removeNestedSessionID(payload, "client_metadata")

	logIDJSON, err := json.Marshal(logID)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ModelHub log ID: %w", err)
	}

	extraJSON, err := json.Marshal(extra)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ModelHub extra metadata: %w", err)
	}

	payload[modelHubLogIDField] = logIDJSON
	payload[modelHubExtraField] = extraJSON

	result, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode ModelHub request body: %w", err)
	}

	return result, nil
}

// addModelHubEnvelope is kept as a small package helper for callers/tests that
// only need to add the native correlation envelope to an already-normalized
// Responses body.
func addModelHubEnvelope(body []byte, logID, extra string) ([]byte, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(bytes.TrimSpace(body), &payload); err != nil {
		return nil, fmt.Errorf("failed to decode ModelHub request body: %w", err)
	}
	if payload == nil {
		return nil, fmt.Errorf("failed to decode ModelHub request body: expected a JSON object")
	}
	deleteJSONKeysFold(payload, modelHubLogIDField, modelHubExtraField)

	logIDJSON, err := json.Marshal(logID)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ModelHub log ID: %w", err)
	}
	extraJSON, err := json.Marshal(extra)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal ModelHub extra metadata: %w", err)
	}
	payload[modelHubLogIDField] = logIDJSON
	payload[modelHubExtraField] = extraJSON

	result, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode ModelHub request body: %w", err)
	}
	return result, nil
}

func normalizeModelHubInput(payload map[string]json.RawMessage) {
	raw, ok := jsonFieldFold(payload, "input")
	if !ok || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return
	}

	var text string
	if json.Unmarshal(raw, &text) != nil {
		return
	}

	input := []map[string]any{{
		"role": "user",
		"content": []map[string]string{{
			"type": "input_text",
			"text": text,
		}},
	}}
	encoded, err := json.Marshal(input)
	if err != nil {
		return
	}
	deleteJSONKeysFold(payload, "input")
	payload["input"] = encoded
}

func normalizeModelHubMaxOutputTokens(payload map[string]json.RawMessage) {
	maxOutput, present := jsonFieldFold(payload, "max_output_tokens")
	if !present || bytes.Equal(bytes.TrimSpace(maxOutput), []byte("null")) {
		if value, ok := jsonFieldFold(payload, "max_completion_tokens"); ok && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			payload["max_output_tokens"] = value
		} else if value, ok := jsonFieldFold(payload, "max_tokens"); ok && !bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			payload["max_output_tokens"] = value
		} else {
			payload["max_output_tokens"] = json.RawMessage(strconv.FormatInt(DefaultMaxOutputTokens, 10))
		}
	}
	if maxOutput, ok := jsonFieldFold(payload, "max_output_tokens"); ok {
		deleteJSONKeysFold(payload, "max_output_tokens")
		payload["max_output_tokens"] = maxOutput
	}
	deleteJSONKeysFold(payload, "max_tokens", "max_completion_tokens")
}

func normalizeModelHubReasoning(payload map[string]json.RawMessage) {
	raw, ok := jsonFieldFold(payload, "reasoning")
	if !ok {
		return
	}

	var reasoning map[string]json.RawMessage
	if json.Unmarshal(raw, &reasoning) != nil || reasoning == nil {
		return
	}
	summary, err := json.Marshal("detailed")
	if err != nil {
		return
	}
	deleteJSONKeysFold(reasoning, "summary")
	reasoning["summary"] = summary
	encoded, err := json.Marshal(reasoning)
	if err != nil {
		return
	}
	deleteJSONKeysFold(payload, "reasoning")
	payload["reasoning"] = encoded
}

func removeNestedSessionID(payload map[string]json.RawMessage, field string) {
	raw, ok := jsonFieldFold(payload, field)
	if !ok {
		return
	}
	nested, ok := rawJSONObject(raw)
	if !ok {
		return
	}
	deleteJSONKeysFold(nested, "session_id")
	if len(nested) == 0 {
		deleteJSONKeysFold(payload, field)
		return
	}
	encoded, err := json.Marshal(nested)
	if err != nil {
		return
	}
	deleteJSONKeysFold(payload, field)
	payload[field] = encoded
}

func resolveSessionID(ctx context.Context, llmReq *llm.Request) string {
	if llmReq != nil && llmReq.RawRequest != nil {
		// Explicit body/session headers win over the server trace fallback. This
		// preserves Codex/compact affinity when tracing is disabled at the gateway.
		if sessionID := sessionIDFromBody(llmReq.RawRequest.Body); sessionID != "" {
			return sessionID
		}
		if sessionID := sessionIDFromHeaders(llmReq.RawRequest.Headers); sessionID != "" {
			return sessionID
		}
	}

	if ctx != nil {
		if sessionID, ok := shared.GetSessionID(ctx); ok {
			if sessionID = validCorrelationID(sessionID); sessionID != "" {
				return sessionID
			}
		}
	}

	if llmReq != nil {
		if llmReq.RawRequest != nil {
			if sessionID := promptCacheKeyFromBody(llmReq.RawRequest.Body); sessionID != "" {
				return sessionID
			}
		}
		if sessionID := validCorrelationID(llmReq.Metadata["session_id"]); sessionID != "" {
			return sessionID
		}
		if llmReq.PromptCacheKey != nil {
			if sessionID := validCorrelationID(*llmReq.PromptCacheKey); sessionID != "" {
				return sessionID
			}
		}
		if llmReq.Compact != nil {
			if sessionID := validCorrelationID(llmReq.Compact.PromptCacheKey); sessionID != "" {
				return sessionID
			}
		}
		if llmReq.TransformerMetadata != nil {
			if sessionID, ok := llmReq.TransformerMetadata[modelHubSessionMetadataKey].(string); ok {
				if sessionID = validCorrelationID(sessionID); sessionID != "" {
					return sessionID
				}
			}
		}
	}

	sessionID := uuid.NewString()
	if llmReq != nil {
		if llmReq.TransformerMetadata == nil {
			llmReq.TransformerMetadata = make(map[string]any)
		}
		llmReq.TransformerMetadata[modelHubSessionMetadataKey] = sessionID
	}
	return sessionID
}

func sessionIDFromHeaders(headers http.Header) string {
	for _, name := range []string{
		"Session-Id",
		"Session_id",
		"X-Session-Id",
		"Thread-Id",
		"X-Trace-Id",
	} {
		if value := validCorrelationID(headerValueFold(headers, name)); value != "" {
			return value
		}
	}

	for _, name := range []string{"X-Codex-Turn-Metadata", modelHubExtraField} {
		if value := sessionIDFromJSONText(headerValueFold(headers, name)); value != "" {
			return value
		}
	}

	return ""
}

func sessionIDFromBody(body []byte) string {
	var payload map[string]json.RawMessage
	if json.Unmarshal(bytes.TrimSpace(body), &payload) != nil || payload == nil {
		return ""
	}

	for _, key := range []string{"session_id", "metadata", modelHubExtraField, "client_metadata"} {
		raw, ok := jsonFieldFold(payload, key)
		if !ok {
			continue
		}
		if key == "session_id" {
			if value := rawJSONString(raw); value != "" {
				return validCorrelationID(value)
			}
			continue
		}
		if key == modelHubExtraField {
			if value := sessionIDFromJSONText(rawJSONString(raw)); value != "" {
				return value
			}
		}
		if nested, ok := rawJSONObject(raw); ok {
			if value, ok := jsonFieldFold(nested, "session_id"); ok {
				if sessionID := validCorrelationID(rawJSONString(value)); sessionID != "" {
					return sessionID
				}
			}
		}
	}
	return ""
}

func promptCacheKeyFromBody(body []byte) string {
	var payload map[string]json.RawMessage
	if json.Unmarshal(bytes.TrimSpace(body), &payload) != nil || payload == nil {
		return ""
	}
	if raw, ok := jsonFieldFold(payload, "prompt_cache_key"); ok {
		return validCorrelationID(rawJSONString(raw))
	}
	return ""
}

func sessionIDFromJSONText(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	var payload map[string]json.RawMessage
	if json.Unmarshal([]byte(value), &payload) != nil || payload == nil {
		return ""
	}
	if raw, ok := jsonFieldFold(payload, "session_id"); ok {
		return validCorrelationID(rawJSONString(raw))
	}
	return ""
}

func jsonFieldFold(payload map[string]json.RawMessage, name string) (json.RawMessage, bool) {
	if payload == nil {
		return nil, false
	}
	if value, ok := payload[name]; ok {
		return value, true
	}
	for key, value := range payload {
		if strings.EqualFold(key, name) {
			return value, true
		}
	}
	return nil, false
}

func rawJSONObject(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, false
	}
	var value map[string]json.RawMessage
	if json.Unmarshal(raw, &value) != nil || value == nil {
		return nil, false
	}
	return value, true
}

func rawJSONString(raw json.RawMessage) string {
	var value string
	if json.Unmarshal(raw, &value) != nil {
		return ""
	}
	return strings.TrimSpace(value)
}

func deleteJSONKeysFold(payload map[string]json.RawMessage, keys ...string) {
	for existing := range payload {
		for _, key := range keys {
			if strings.EqualFold(existing, key) {
				delete(payload, existing)
				break
			}
		}
	}
}

func headerValueFold(headers http.Header, name string) string {
	if headers == nil {
		return ""
	}
	if value := headers.Get(name); value != "" {
		return value
	}
	for key, values := range headers {
		if strings.EqualFold(key, name) && len(values) > 0 {
			return values[0]
		}
	}
	return ""
}

func validCorrelationID(value string) string {
	value = strings.TrimSpace(value)
	if strings.ContainsAny(value, "\r\n") {
		return ""
	}
	return value
}
