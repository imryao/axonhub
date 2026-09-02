package orchestrator

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

func TestExtractErrorMessageUsesRetainedHTTPBody(t *testing.T) {
	rawErr := &httpclient.Error{
		StatusCode: http.StatusBadRequest,
		Status:     "400 Bad Request",
		URL:        "https://provider.example/v1/responses?api_key=secret",
		Body:       []byte(`{"code":-4201,"message":"invalid_encrypted_content"}`),
	}
	publicErr := &llm.ResponseError{StatusCode: http.StatusBadRequest}
	err := pipeline.WrapUpstreamErrorWithRaw(publicErr, rawErr)

	message := ExtractErrorMessage(err)
	require.Contains(t, message, "invalid_encrypted_content")
	require.NotContains(t, message, "api_key=secret")
	require.Equal(t, http.StatusBadRequest, *ExtractErrorInfo(err).StatusCode)
}

func TestExtractErrorMessageUsesRawResponsesEvent(t *testing.T) {
	raw := []byte(`{"type":"error","error":{"code":"invalid_encrypted_content","message":"encrypted content could not be verified"}}`)
	err := &llm.ResponseError{
		StatusCode: http.StatusBadRequest,
		Detail:     llm.ErrorDetail{Type: "stream_error", Message: "stream error"},
		RawBody:    raw,
	}

	message := ExtractErrorMessage(err)
	require.Contains(t, message, "encrypted content could not be verified")
	require.Contains(t, message, "upstream_event")
	require.Equal(t, http.StatusBadRequest, *ExtractErrorInfo(err).StatusCode)
}

func TestSanitizeUpstreamURLDropsQueryAndFragment(t *testing.T) {
	require.Equal(t, "https://provider.example/v1/responses", sanitizeUpstreamURL("https://provider.example/v1/responses?api_key=secret#fragment"))
}

func TestSanitizeResponseBodyRedactsCredentialFields(t *testing.T) {
	body := []byte(`{"authorization":"secret","encrypted_content":"opaque","message":"keep this"}`)
	sanitized := string(sanitizeResponseBody(body, 1024))
	require.NotContains(t, sanitized, "secret")
	require.NotContains(t, sanitized, "opaque")
	require.Contains(t, sanitized, "keep this")
}
