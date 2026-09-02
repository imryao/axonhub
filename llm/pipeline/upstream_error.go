package pipeline

import "errors"

// UpstreamError marks errors that originate from the upstream provider path.
type UpstreamError struct {
	Err error

	// RawErr keeps the provider-facing error before it is transformed into the
	// public/unified error shape.  In particular, HTTP transformers often move
	// the response body into llm.ResponseError; retaining the original error lets
	// the server log and persist the provider's diagnostic payload without
	// exposing it to the API client when upstream-error redaction is enabled.
	RawErr error
}

func (e *UpstreamError) Error() string {
	if e == nil || e.Err == nil {
		return "upstream error"
	}

	return e.Err.Error()
}

func (e *UpstreamError) Unwrap() error {
	if e == nil {
		return nil
	}

	return e.Err
}

// RawError returns the original provider-facing error, when one was retained
// by WrapUpstreamErrorWithRaw.  It intentionally does not participate in the
// regular Unwrap chain: callers that handle the public error type should keep
// seeing the transformed error, while diagnostics can opt into the raw value.
func (e *UpstreamError) RawError() error {
	if e == nil {
		return nil
	}

	if e.RawErr != nil {
		return e.RawErr
	}

	// Preserve diagnostics when wrappers are layered (for example, a retry
	// wrapper around an already-marked upstream error).
	return RawError(e.Err)
}

func WrapUpstreamError(err error) error {
	if err == nil {
		return nil
	}

	var upstreamErr *UpstreamError
	if errors.As(err, &upstreamErr) {
		return err
	}

	return &UpstreamError{Err: err}
}

// WrapUpstreamErrorWithRaw marks err as an upstream failure while retaining
// the original provider-facing error for diagnostics.  The raw value is kept
// out of Error/Unwrap so API error transformation and retry classification keep
// their existing semantics.
func WrapUpstreamErrorWithRaw(err, rawErr error) error {
	if err == nil {
		return nil
	}

	if rawErr == nil {
		return WrapUpstreamError(err)
	}

	var existing *UpstreamError
	if errors.As(err, &existing) && existing != nil && existing.RawErr != nil {
		return err
	}

	return &UpstreamError{Err: err, RawErr: rawErr}
}

// RawError extracts a provider-facing error retained by an upstream wrapper.
// It returns nil when the error chain has no raw diagnostic value.
func RawError(err error) error {
	if err == nil {
		return nil
	}

	var carrier interface{ RawError() error }
	if errors.As(err, &carrier) {
		if rawErr := carrier.RawError(); rawErr != nil {
			return rawErr
		}
	}

	return nil
}

func IsUpstreamError(err error) bool {
	var upstreamErr *UpstreamError
	return errors.As(err, &upstreamErr)
}
