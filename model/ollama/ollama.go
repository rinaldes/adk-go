package ollama

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"runtime"
	"strings"

	"google.golang.org/genai"

	"github.com/rinaldes/adk-go/internal/llminternal"
	"github.com/rinaldes/adk-go/internal/llminternal/converters"
	"github.com/rinaldes/adk-go/internal/llminternal/googlellm"
	"github.com/rinaldes/adk-go/internal/version"
	"github.com/rinaldes/adk-go/model"
)

type geminiModel struct {
	client             *genai.Client
	name               string
	versionHeaderValue string
}

func NewModel(ctx context.Context, modelName string, cfg *genai.ClientConfig) (model.LLM, error) {
	if cfg == nil {
		cfg = &genai.ClientConfig{}
	} else {
		cfgCopy := *cfg
		if cfg.HTTPClient != nil {
			clientCopy := *cfg.HTTPClient
			cfgCopy.HTTPClient = &clientCopy
		}
		cfg = &cfgCopy
	}

	cfg.Backend = genai.BackendGeminiAPI
	cfg.APIKey = "ollama"
	if cfg.HTTPOptions.BaseURL == "" {
		cfg.HTTPOptions.BaseURL = ollamaBaseURL()
	}

	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, err
	}

	if client.ClientConfig().HTTPClient != nil {
		client.ClientConfig().HTTPClient.Transport = &debugTransport{
			base: client.ClientConfig().HTTPClient.Transport,
		}
	}

	headerValue := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version,
		strings.TrimPrefix(runtime.Version(), "go"))

	return &geminiModel{
		name:               modelName,
		client:             client,
		versionHeaderValue: headerValue,
	}, nil
}

func ollamaBaseURL() string {
	if u := os.Getenv("OLLAMA_BASE_URL"); u != "" {
		return u
	}
	return "http://localhost:11434/v1"
}

func (m *geminiModel) Name() string { return m.name }

func (m *geminiModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}
	m.addHeaders(req.Config.HTTPOptions.Headers)

	if stream {
		return m.generateStream(ctx, req)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, req)
		yield(resp, err)
	}
}

func (m *geminiModel) addHeaders(headers http.Header) {
	headers.Set("x-goog-api-client", m.versionHeaderValue)
	headers.Set("user-agent", m.versionHeaderValue)
}

func (m *geminiModel) modelName(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}
	return m.name
}

func (m *geminiModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	resp, err := m.client.Models.GenerateContent(ctx, m.modelName(req), req.Contents, req.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("empty response")
	}
	return converters.Genai2LLMResponse(resp), nil
}

func (m *geminiModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()
	return func(yield func(*model.LLMResponse, error) bool) {
		for resp, err := range m.client.Models.GenerateContentStream(ctx, m.modelName(req), req.Contents, req.Config) {
			if err != nil {
				yield(nil, err)
				return
			}
			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

func (m *geminiModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}
	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

type mergeHeadersInterceptor struct {
	base http.RoundTripper
}

type debugTransport struct {
	base http.RoundTripper
}

func (d *debugTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	fmt.Printf(">>> SDK is calling: %s %s\n", req.Method, req.URL.String())
	if d.base == nil {
		return http.DefaultTransport.RoundTrip(req)
	}
	return d.base.RoundTrip(req)
}

func (m *geminiModel) GetGoogleLLMVariant() genai.Backend {
	if m == nil || m.client == nil {
		return genai.BackendUnspecified
	}
	return m.client.ClientConfig().Backend
}

var _ googlellm.GoogleLLM = &geminiModel{}
