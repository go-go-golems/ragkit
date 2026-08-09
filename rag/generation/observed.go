package generation

import (
	"context"
	"time"

	"github.com/go-go-golems/ragkit/rag"
	"github.com/pkg/errors"
)

// GenerationObservation records one provider call. Request and response
// payloads are present only when enabled by ObservationPolicy.
type GenerationObservation struct {
	StartedAt     time.Time              `json:"started_at"`
	Duration      time.Duration          `json:"duration"`
	RequestKind   string                 `json:"request_kind"`
	Model         string                 `json:"model"`
	FinishReason  string                 `json:"finish_reason,omitempty"`
	Usage         rag.Usage              `json:"usage"`
	Request       *rag.GenerationRequest `json:"request,omitempty"`
	Response      *rag.GenerationResult  `json:"response,omitempty"`
	ProviderError string                 `json:"provider_error,omitempty"`
}

// ObservationPolicy explicitly authorizes provider payload disclosure.
// OutputSchema follows IncludePrompt because both are generation instructions.
type ObservationPolicy struct {
	IncludePrompt   bool
	IncludeText     bool
	IncludeEvidence bool
	IncludeResponse bool
}

// GenerationObserver persists or otherwise consumes one provider observation.
type GenerationObserver func(context.Context, GenerationObservation) error

type observedGenerator struct {
	generator rag.Generator
	policy    ObservationPolicy
	observer  GenerationObserver
}

var _ rag.Generator = (*observedGenerator)(nil)

// ObservedGenerator decorates a generator with provider-call observation.
func ObservedGenerator(
	generator rag.Generator,
	policy ObservationPolicy,
	observer GenerationObserver,
) (rag.Generator, error) {
	if generator == nil {
		return nil, errors.New("observed generator requires a generator")
	}
	if observer == nil {
		return nil, errors.New("observed generator requires an observer")
	}
	return &observedGenerator{generator: generator, policy: policy, observer: observer}, nil
}

func (generator *observedGenerator) Generate(
	ctx context.Context,
	request rag.GenerationRequest,
) (rag.GenerationResult, error) {
	startedAt := time.Now().UTC()
	result, providerErr := generator.generator.Generate(ctx, request)
	observation := GenerationObservation{
		StartedAt: startedAt, Duration: time.Since(startedAt),
		RequestKind: request.Kind, Model: request.Model,
		FinishReason: result.FinishReason, Usage: result.Usage,
		Request: disclosedRequest(request, generator.policy),
	}
	if generator.policy.IncludeResponse && providerErr == nil {
		response := result
		observation.Response = &response
	}
	if providerErr != nil {
		observation.ProviderError = providerErr.Error()
	}
	if err := generator.observer(context.WithoutCancel(ctx), observation); err != nil {
		return result, errors.Wrap(err, "observe generation provider call")
	}
	if providerErr != nil {
		return result, providerErr
	}
	return result, nil
}

func disclosedRequest(
	request rag.GenerationRequest,
	policy ObservationPolicy,
) *rag.GenerationRequest {
	if !policy.IncludePrompt && !policy.IncludeText && !policy.IncludeEvidence {
		return nil
	}
	disclosed := rag.GenerationRequest{Kind: request.Kind, Model: request.Model}
	if policy.IncludePrompt {
		disclosed.Prompt = request.Prompt
		disclosed.OutputSchema = request.OutputSchema
	}
	if policy.IncludeText {
		disclosed.Text = request.Text
	}
	if policy.IncludeEvidence {
		disclosed.Evidence = append([]rag.Evidence(nil), request.Evidence...)
	}
	return &disclosed
}
