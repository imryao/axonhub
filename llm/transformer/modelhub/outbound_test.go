package modelhub

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestNewOutboundTransformerDefaultsAndRejectsURLCredentials(t *testing.T) {
	tr, err := NewOutboundTransformer("", "test-ak")
	require.NoError(t, err)
	require.NotNil(t, tr)

	_, err = NewOutboundTransformer("https://example.test/modelhub?ak=embedded", "test-ak")
	require.Error(t, err)

	_, err = NewOutboundTransformer("https://example.test/modelhub#fragment", "test-ak")
	require.Error(t, err)

	err = ValidateBaseURL("https://example.test/modelhub?ak=secret")
	require.Error(t, err)
	require.NotContains(t, err.Error(), "secret")
}

func TestTransformRequestAddsModelHubAuthAndEnvelope(t *testing.T) {
	tr, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL:        "https://example.test/api/modelhub/online",
		APIKeyProvider: auth.NewStaticKeyProvider("test-ak"),
	})
	require.NoError(t, err)

	llmReq := &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeChat,
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
	}

	httpReq, err := tr.TransformRequest(shared.WithSessionID(context.Background(), "session-123"), llmReq)
	require.NoError(t, err)
	require.Equal(t, "https://example.test/api/modelhub/online/responses", httpReq.URL)
	require.Equal(t, "test-ak", httpReq.Query.Get(APIKeyQueryParameter))
	require.Equal(t, []string{APIKeyQueryParameter}, httpReq.SensitiveQueryParameters)
	require.True(t, httpReq.SkipInboundQueryMerge)
	require.Nil(t, httpReq.Auth)
	require.Empty(t, httpReq.Headers.Get("Authorization"))
	require.NotEmpty(t, httpReq.Headers.Get("X-TT-LOGID"))
	require.Equal(t, `{"session_id":"session-123"}`, httpReq.Headers.Get("Extra"))

	var body map[string]any
	require.NoError(t, json.Unmarshal(httpReq.Body, &body))
	require.Equal(t, httpReq.Headers.Get("X-TT-LOGID"), body["X-TT-logid"])
	require.Equal(t, `{"session_id":"session-123"}`, body["extra"])
}

func TestTransformRequestCompactUsesModelHubCompactPath(t *testing.T) {
	tr, err := NewOutboundTransformer("https://example.test/api/modelhub/online", "test-ak")
	require.NoError(t, err)

	llmReq := &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeCompact,
		Compact: &llm.CompactRequest{
			Input: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: lo.ToPtr("compact")}}},
		},
	}

	httpReq, err := tr.TransformRequest(context.Background(), llmReq)
	require.NoError(t, err)
	require.Equal(t, "https://example.test/api/modelhub/online/responses/compact", httpReq.URL)
	require.Equal(t, "test-ak", httpReq.Query.Get(APIKeyQueryParameter))
}

func TestTransformRequestRequiresAPIKey(t *testing.T) {
	tr, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL:        "https://example.test/api/modelhub/online",
		APIKeyProvider: auth.NewStaticKeyProvider(""),
	})
	require.NoError(t, err)

	_, err = tr.TransformRequest(context.Background(), &llm.Request{Model: "gpt-5.6-sol"})
	require.Error(t, err)
	require.True(t, strings.Contains(err.Error(), "API key"))
}

func TestTransformRequestURLQueryEncodesModelHubAK(t *testing.T) {
	tr, err := NewOutboundTransformer("https://example.test/api/modelhub/online", "ak value/with?chars")
	require.NoError(t, err)
	req, err := tr.TransformRequest(context.Background(), &llm.Request{
		Model: "gpt-5.6-sol",
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
	})
	require.NoError(t, err)
	require.Equal(t, "ak value/with?chars", req.Query.Get(APIKeyQueryParameter))
	require.Equal(t, "ak=ak+value%2Fwith%3Fchars", req.Query.Encode())
}

func TestTransformRequestNormalizesModelHubResponsesPayload(t *testing.T) {
	tr, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL:        "https://example.test/api/modelhub/online",
		APIKeyProvider: auth.NewStaticKeyProvider("test-ak"),
	})
	require.NoError(t, err)

	stream := true
	llmReq := &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeChat,
		Stream:      &stream,
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
		RawRequest: &httpclient.Request{Headers: http.Header{
			"Session-Id": {"header-session"},
		}},
	}

	httpReq, err := tr.TransformRequest(context.Background(), llmReq)
	require.NoError(t, err)

	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(httpReq.Body, &body))
	require.Equal(t, "true", string(body["stream"]))
	require.Equal(t, "128000", string(body["max_output_tokens"]))
	require.NotEqual(t, `"hello"`, string(body["input"]))
	var extra string
	require.NoError(t, json.Unmarshal(body["extra"], &extra))
	require.Equal(t, `{"session_id":"header-session"}`, extra)

	for _, key := range []string{
		"metadata",
		"parallel_tool_calls",
		"previous_response_id",
		"prompt_cache_key",
		"service_tier",
		"store",
		"stream_options",
		"user",
	} {
		_, present := body[key]
		require.False(t, present, "%s should not be sent to ModelHub", key)
	}
}

func TestTransformRequestUsesCompactSessionAndCustomPath(t *testing.T) {
	tr, err := NewOutboundTransformerWithConfig(&Config{
		BaseURL:        "https://example.test/api/modelhub/online",
		EndpointPath:   "/compact",
		APIKeyProvider: auth.NewStaticKeyProvider("test-ak"),
	})
	require.NoError(t, err)

	llmReq := &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeCompact,
		Compact:     &llm.CompactRequest{},
		RawRequest:  &httpclient.Request{Body: []byte(`{"model":"client-model","session_id":"compact-session","parallel_tool_calls":true}`)},
	}

	httpReq, err := tr.TransformRequest(context.Background(), llmReq)
	require.NoError(t, err)
	require.Equal(t, "https://example.test/api/modelhub/online/compact", httpReq.URL)

	var body map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(httpReq.Body, &body))
	var extra string
	require.NoError(t, json.Unmarshal(body["extra"], &extra))
	require.Equal(t, `{"session_id":"compact-session"}`, extra)
	require.Equal(t, "128000", string(body["max_output_tokens"]))
	require.Equal(t, `"gpt-5.6-sol"`, string(body["model"]))
	require.Equal(t, "true", string(body["parallel_tool_calls"]))
}

func TestTransformRequestKeepsGeneratedSessionAcrossRetries(t *testing.T) {
	tr, err := NewOutboundTransformer("https://example.test/api/modelhub/online", "test-ak")
	require.NoError(t, err)
	req := &llm.Request{
		Model:       "gpt-5.6-sol",
		RequestType: llm.RequestTypeChat,
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
	}

	first, err := tr.TransformRequest(context.Background(), req)
	require.NoError(t, err)
	second, err := tr.TransformRequest(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, first.Headers.Get("Extra"), second.Headers.Get("Extra"))
}

func TestTransformRequestPrefersClientSessionHeaderOverContext(t *testing.T) {
	tr, err := NewOutboundTransformer("https://example.test", "test-ak")
	require.NoError(t, err)
	llmReq := &llm.Request{
		Model: "gpt-5.6-sol",
		Messages: []llm.Message{{
			Role:    "user",
			Content: llm.MessageContent{Content: lo.ToPtr("hello")},
		}},
		RawRequest: &httpclient.Request{Headers: http.Header{"Session-Id": {"header-session"}}},
	}
	httpReq, err := tr.TransformRequest(shared.WithSessionID(context.Background(), "context-session"), llmReq)
	require.NoError(t, err)
	require.Equal(t, `{"session_id":"header-session"}`, httpReq.Headers.Get("Extra"))
}

func TestModelHubDisablesBodyPassThrough(t *testing.T) {
	tr, err := NewOutboundTransformer("https://example.test", "test-ak")
	require.NoError(t, err)
	require.False(t, tr.(*OutboundTransformer).AllowPassThroughBody(context.Background(), nil, nil))
}

func TestPrepareModelHubBodyRejectsNonObject(t *testing.T) {
	_, err := prepareModelHubBody([]byte("null"), &llm.Request{}, "log", `{"session_id":"session"}`)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expected a JSON object")
}

func TestRedactModelHubURL(t *testing.T) {
	redacted := redactModelHubURL("https://example.test/responses?ak=secret&region=cn")
	require.NotContains(t, redacted, "secret")
	require.Contains(t, redacted, "ak=%3Credacted%3E")
	require.Contains(t, redacted, "region=cn")
	malformed := redactModelHubURL("https://example.test/responses?ak=secret%zz")
	require.NotContains(t, malformed, "secret")
}

func TestModelHubCorrelationHeadersAreNotOverriddenByInboundHeaders(t *testing.T) {
	request := &httpclient.Request{Headers: http.Header{
		"X-Tt-Logid": {"server-log"},
		"Extra":      {`{"session_id":"server-session"}`},
	}}
	inbound := &httpclient.Request{Headers: http.Header{
		"X-TT-LOGID": {"client-log"},
		"extra":      {`{"session_id":"client-session"}`},
	}}

	httpclient.MergeInboundRequest(request, inbound)
	require.Equal(t, "server-log", request.Headers.Get("X-TT-LOGID"))
	require.Equal(t, `{"session_id":"server-session"}`, request.Headers.Get("Extra"))
}
