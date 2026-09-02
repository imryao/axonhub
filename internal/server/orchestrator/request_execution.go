package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/tidwall/gjson"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/internal/pkg/xcontext"
	"github.com/looplj/axonhub/internal/pkg/xerrors"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

// Precompiled regex patterns for sanitizeResponseBody to avoid recompiling on each call.
var (
	tokenRegex       = regexp.MustCompile(`(?i)(bearer[\s:=]+)[a-zA-Z0-9_\-\.]+`)
	apiKeyRegex      = regexp.MustCompile(`(api[keyK]ey|API[keyK]ey)["']?\s*[:=]\s*["']?([a-zA-Z0-9_\-\.]{8,})["']?`)
	secretFieldRegex = regexp.MustCompile(`(?i)("(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|cookie|encrypted_content)"\s*:\s*")[^"]*(")`)
	emailRegex       = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)
)

// sanitizeResponseBody redacts obvious secrets and truncates the body for safe logging.
func sanitizeResponseBody(body []byte, maxLen int) []byte {
	if len(body) == 0 {
		return body
	}

	str := string(body)

	// Redact bearer tokens (case-insensitive), preserving the bearer prefix
	str = tokenRegex.ReplaceAllString(str, "${1}[REDACTED]")

	// Redact API keys (common patterns)
	str = apiKeyRegex.ReplaceAllString(str, "$1=[REDACTED]")

	// Redact common JSON credential fields, including account-bound encrypted
	// reasoning blobs that should never be copied into application logs.
	str = secretFieldRegex.ReplaceAllString(str, "${1}[REDACTED]${2}")

	// Redact email addresses
	str = emailRegex.ReplaceAllString(str, "[EMAIL REDACTED]")

	// Truncate if too long
	if len(str) > maxLen {
		str = str[:maxLen] + "..."
	}

	return []byte(str)
}

// persistRequestExecutionMiddleware ensures a request execution exists and handles error updates.
type persistRequestExecutionMiddleware struct {
	pipeline.DummyMiddleware

	outbound *PersistentOutboundTransformer

	rawResponse *httpclient.Response
}

func persistRequestExecution(outbound *PersistentOutboundTransformer) pipeline.Middleware {
	return &persistRequestExecutionMiddleware{
		outbound: outbound,
	}
}

func (m *persistRequestExecutionMiddleware) Name() string {
	return "persist-request-execution"
}

func (m *persistRequestExecutionMiddleware) OnOutboundRawRequest(ctx context.Context, request *httpclient.Request) (*httpclient.Request, error) {
	state := m.outbound.state
	if state == nil || state.RequestExec != nil {
		return request, nil
	}

	channel := m.outbound.GetCurrentChannel()
	if channel == nil {
		return request, nil
	}

	candidate := state.ChannelModelsCandidates[state.CurrentCandidateIndex]
	entry := candidate.Models[state.CurrentModelIndex]

	// Prefer the API format of the actual outbound request: transformers may emit
	// multiple formats (e.g. OpenAI outbound also builds audio speech/transcription
	// requests) while APIFormat() only reports the primary one.
	format := m.outbound.APIFormat()
	if request.APIFormat != "" {
		format = llm.APIFormat(request.APIFormat)
	}

	requestExec, err := state.RequestService.CreateRequestExecution(
		ctx,
		channel,
		entry.ActualModel,
		state.Request,
		*request,
		format,
		state.PassThroughApplied,
	)
	if err != nil {
		return nil, err
	}

	// Update request with channel ID after channel selection
	if state.Request != nil && state.Request.ChannelID != channel.ID {
		err := state.RequestService.UpdateRequestChannelID(ctx, state.Request.ID, channel.ID)
		if err != nil {
			return nil, err
		}
		// Update the in-memory state to prevent duplicate updates and ensure consistency
		state.Request.ChannelID = channel.ID
	}

	state.RequestExec = requestExec

	return request, nil
}

func (m *persistRequestExecutionMiddleware) OnOutboundRawResponse(ctx context.Context, response *httpclient.Response) (*httpclient.Response, error) {
	m.rawResponse = response
	return response, nil
}

func (m *persistRequestExecutionMiddleware) OnOutboundLlmResponse(ctx context.Context, llmResp *llm.Response) (*llm.Response, error) {
	state := m.outbound.state
	if state == nil || state.RequestExec == nil {
		return llmResp, nil
	}

	// Use context without cancellation to ensure persistence even if client canceled
	persistCtx, cancel := xcontext.DetachWithTimeout(ctx, 10*time.Second)
	defer cancel()

	// Build latency metrics from performance record
	var metrics *biz.LatencyMetrics

	if state.Perf != nil && !state.Perf.StartTime.IsZero() {
		var (
			firstTokenLatencyMs int64
			requestLatencyMs    int64
		)

		if state.Perf.RequestCompleted && !state.Perf.EndTime.IsZero() {
			firstTokenLatencyMs, requestLatencyMs, _ = state.Perf.Calculate()
		} else {
			requestLatencyMs = time.Since(state.Perf.StartTime).Milliseconds()
			if state.Perf.Stream && state.Perf.FirstTokenTime != nil {
				firstTokenLatencyMs = state.Perf.FirstTokenTime.Sub(state.Perf.StartTime).Milliseconds()
			}

			requestLatencyMs = biz.ClampLatency(requestLatencyMs)
			firstTokenLatencyMs = biz.ClampLatency(firstTokenLatencyMs)
		}

		metrics = &biz.LatencyMetrics{
			LatencyMs: &requestLatencyMs,
		}
		if state.Perf.Stream && state.Perf.FirstTokenTime != nil {
			metrics.FirstTokenLatencyMs = &firstTokenLatencyMs
		}

		if state.Perf.Stream {
			reasoningDurationMs := state.Perf.CalculateReasoningDurationMs()
			if reasoningDurationMs > 0 {
				metrics.ReasoningDurationMs = &reasoningDurationMs
			}
		}
	}

	// Audio responses (binary TTS / non-JSON STT) must be converted to JSON-safe payloads
	// before persisting into the JSON response_body column.
	respBody := audioSafeResponseBody(llmResp.RequestType, m.rawResponse.Headers.Get("Content-Type"), m.rawResponse.Body)

	err := state.RequestService.UpdateRequestExecutionCompleted(
		persistCtx,
		state.RequestExec.ID,
		llmResp.ID,
		respBody,
		metrics,
	)
	if err != nil {
		log.Warn(persistCtx, "Failed to update request execution status to completed", log.Cause(err))
	}

	return llmResp, nil
}

func (m *persistRequestExecutionMiddleware) OnOutboundRawError(ctx context.Context, err error) {
	// Update request execution with the real error message when request fails
	state := m.outbound.state
	if state == nil || state.RequestExec == nil {
		return
	}

	// Log error with channel information for better debugging
	channel := m.outbound.GetCurrentChannel()
	if channel != nil {
		logFields := []log.Field{
			log.Cause(err),
			log.Int("channel_id", channel.ID),
			log.String("channel_name", channel.Name),
		}
		if modelID := m.outbound.GetCurrentModelID(); modelID != "" {
			logFields = append(logFields, log.String("model_id", modelID))
		}
		logFields = appendUpstreamErrorDiagnostics(logFields, err)

		log.Warn(ctx, "request process failed", logFields...)
	}

	// Use context without cancellation to ensure persistence even if client canceled
	persistCtx, cancel := xcontext.DetachWithTimeout(ctx, 10*time.Second)
	defer cancel()

	updateErr := state.RequestService.UpdateRequestExecutionFailed(
		persistCtx,
		state.RequestExec.ID,
		ExtractErrorMessage(err),
		ExtractErrorInfo(err),
	)
	if updateErr != nil {
		log.Warn(persistCtx, "Failed to update request execution status to failed", log.Cause(updateErr))
	}
}

// ExtractErrorInfo extracts HTTP status code and sanitized response body from error.
func ExtractErrorInfo(err error) *biz.ExecutionErrorInfo {
	httpErr, ok := xerrors.As[*httpclient.Error](diagnosticError(err))
	if ok && httpErr != nil {
		return &biz.ExecutionErrorInfo{
			StatusCode: &httpErr.StatusCode,
		}
	}

	var responseErr *llm.ResponseError
	if errors.As(err, &responseErr) && responseErr != nil && responseErr.StatusCode != 0 {
		return &biz.ExecutionErrorInfo{
			StatusCode: &responseErr.StatusCode,
		}
	}

	return nil
}

// ExtractErrorMessage extracts HTTP error message from error.
func ExtractErrorMessage(err error) string {
	if err == nil {
		return ""
	}

	// HTTP errors are transformed before the pipeline returns to the
	// orchestrator. Prefer the retained provider error so its response body is
	// still available for persistence and diagnostics.
	diagnosticErr := diagnosticError(err)
	if httpErr, ok := xerrors.As[*httpclient.Error](diagnosticErr); ok {
		return extractHTTPErrorMessage(httpErr)
	}

	var responseErr *llm.ResponseError
	if errors.As(err, &responseErr) && responseErr != nil {
		message := strings.TrimSpace(responseErr.Error())
		if len(responseErr.RawBody) > 0 {
			raw := strings.TrimSpace(string(sanitizeResponseBody(responseErr.RawBody, errorMatchBodyLimit)))
			if raw != "" && !strings.Contains(message, raw) {
				if message == "" || message == "upstream response error" {
					return raw
				}

				return message + "; upstream_event: " + raw
			}
		}
		if message != "" {
			return message
		}
	}

	message := strings.TrimSpace(err.Error())
	if message != "" {
		return message
	}

	return "upstream error (no diagnostic message)"
}

func extractHTTPErrorMessage(httpErr *httpclient.Error) string {
	if httpErr == nil {
		return "upstream HTTP error"
	}

	// Anthropic && OpenAI error format.
	message := gjson.GetBytes(httpErr.Body, "error.message")
	if message.Exists() && message.Type == gjson.String && strings.TrimSpace(message.String()) != "" {
		return message.String()
	}

	// Other compatible error format.
	// Try errors.0.message first, then fall back to errors.message
	message1 := gjson.GetBytes(httpErr.Body, "errors.0.message")
	message2 := gjson.GetBytes(httpErr.Body, "errors.message")

	if message1.Exists() && message1.Type == gjson.String && strings.TrimSpace(message1.String()) != "" {
		return message1.String()
	}

	if message2.Exists() && message2.Type == gjson.String && strings.TrimSpace(message2.String()) != "" {
		return message2.String()
	}

	// Keep the raw provider body when no standard message field exists. This is
	// what makes gateway-specific errors (for example code -4201 wrappers)
	// diagnosable instead of reducing them to a generic status line.
	if body := strings.TrimSpace(string(sanitizeResponseBody(httpErr.Body, errorMatchBodyLimit))); body != "" {
		return body
	}

	if status := strings.TrimSpace(httpErr.Status); status != "" {
		return fmt.Sprintf("upstream HTTP error: %s", status)
	}

	if httpErr.StatusCode != 0 {
		return fmt.Sprintf("upstream HTTP error: status %d", httpErr.StatusCode)
	}

	return "upstream HTTP error"
}

// diagnosticError returns the provider-facing error retained by the pipeline,
// if any. The public error remains transformed so upstream-error redaction and
// retry classification keep their existing behavior.
func diagnosticError(err error) error {
	if rawErr := pipeline.RawError(err); rawErr != nil {
		return rawErr
	}

	return err
}

// appendUpstreamErrorDiagnostics adds bounded, sanitized provider details to
// the request failure log. Keep this separate from Error() so public API
// responses can still hide provider details according to policy.
func appendUpstreamErrorDiagnostics(fields []log.Field, err error) []log.Field {
	if err == nil {
		return fields
	}

	providerErr := diagnosticError(err)
	fields = append(fields,
		log.String("error_type", fmt.Sprintf("%T", providerErr)),
	)

	if httpErr, ok := xerrors.As[*httpclient.Error](providerErr); ok && httpErr != nil {
		fields = append(fields,
			log.Int("upstream_status_code", httpErr.StatusCode),
			log.String("upstream_status", httpErr.Status),
			log.String("upstream_url", sanitizeUpstreamURL(httpErr.URL)),
		)
		if body := sanitizeResponseBody(httpErr.Body, errorMatchBodyLimit); len(body) > 0 {
			// Keep the historical field name for existing log queries and add an
			// explicit upstream-prefixed alias for new consumers.
			fields = append(fields,
				log.ByteString("response_body", body),
				log.ByteString("upstream_response_body", body),
			)
		}
	}

	var responseErr *llm.ResponseError
	if errors.As(err, &responseErr) && responseErr != nil {
		fields = append(fields,
			log.Int("upstream_error_status_code", responseErr.StatusCode),
			log.String("upstream_error_code", responseErr.Detail.Code),
			log.String("upstream_error_type", responseErr.Detail.Type),
			log.ByteString("upstream_error_message", sanitizeResponseBody([]byte(responseErr.Detail.Message), errorMatchBodyLimit)),
			log.String("upstream_error_request_id", responseErr.Detail.RequestID),
		)
		if responseErr.RawEventType != "" {
			fields = append(fields, log.String("upstream_event_type", responseErr.RawEventType))
		}
		if body := sanitizeResponseBody(responseErr.RawBody, errorMatchBodyLimit); len(body) > 0 {
			fields = append(fields, log.ByteString("upstream_event_body", body))
		}
	}

	return fields
}

func sanitizeUpstreamURL(rawURL string) string {
	if rawURL == "" {
		return ""
	}

	// Request URLs may contain provider API keys in query parameters. Keep the
	// host/path useful for diagnosis while dropping the query and fragment.
	if idx := strings.IndexByte(rawURL, '?'); idx >= 0 {
		rawURL = rawURL[:idx]
	}
	if idx := strings.IndexByte(rawURL, '#'); idx >= 0 {
		rawURL = rawURL[:idx]
	}

	return rawURL
}
