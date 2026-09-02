package pipeline

import (
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestWrapUpstreamErrorWithRawPreservesPublicAndRawErrors(t *testing.T) {
	publicErr := &llm.ResponseError{Detail: llm.ErrorDetail{Message: "provider error"}}
	rawErr := &httpclient.Error{
		StatusCode: http.StatusBadRequest,
		Status:     "400 Bad Request",
		Body:       []byte(`{"error":{"code":"provider_code"}}`),
	}

	err := WrapUpstreamErrorWithRaw(publicErr, rawErr)
	require.Equal(t, "error: provider error", err.Error())
	require.ErrorIs(t, err, publicErr)
	require.True(t, IsUpstreamError(err))
	require.Same(t, rawErr, RawError(err))

	var gotPublic *llm.ResponseError
	require.True(t, errors.As(err, &gotPublic))
	require.Same(t, publicErr, gotPublic)

	var gotRaw *httpclient.Error
	require.False(t, errors.As(err, &gotRaw), "raw provider error must stay out of the public unwrap chain")
}

func TestWrapUpstreamErrorWithRawFallsBackToRegularWrapper(t *testing.T) {
	base := errors.New("base")
	err := WrapUpstreamErrorWithRaw(base, nil)
	require.Equal(t, "base", err.Error())
	require.ErrorIs(t, err, base)
	require.Nil(t, RawError(err))
}
