// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ollama implements the [model.LLM] interface for Ollama models.
// It uses the genai client configured to talk to Ollama's OpenAI-compatible API.
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

// TODO: test coverage
type ollamaModel struct {
	client             *genai.Client
	name               string
	versionHeaderValue string
}

// NewModel returns [model.LLM], backed by Ollama's API.
//
// It uses the provided context and configuration to initialize the underlying
// [genai.Client] pointing to Ollama's OpenAI-compatible endpoint.
// The modelName specifies which Ollama model to target (e.g., "llama3.2").
//
// An error is returned if the [genai.Client] fails to initialize.
func NewModel(ctx context.Context, modelName string, cfg *genai.ClientConfig) (model.LLM, error) {
	// Create a copy of the config to avoid mutating the caller's config
	// or the underlying http.Client.
	if cfg == nil {
		cfg = &genai.ClientConfig{}
	}
	cfgCopy := *cfg
	if cfg.HTTPClient != nil {
		clientCopy := *cfg.HTTPClient
		cfgCopy.HTTPClient = &clientCopy
	}

	// Configure for Ollama
	cfgCopy.Backend = genai.BackendGeminiAPI
	cfgCopy.APIKey = "ollama"
	if cfgCopy.HTTPOptions.BaseURL == "" {
		cfgCopy.HTTPOptions.BaseURL = ollamaBaseURL()
	}
	cfg = &cfgCopy

	client, err := genai.NewClient(ctx, cfg)
	if err != nil {
		return nil, err
	}

	if client.ClientConfig().HTTPClient != nil {
		client.ClientConfig().HTTPClient.Transport = &mergeHeadersInterceptor{
			base: client.ClientConfig().HTTPClient.Transport,
		}
	}

	// Create header value once, when the model is created
	headerValue := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version,
		strings.TrimPrefix(runtime.Version(), "go"))

	return &ollamaModel{
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

func (m *ollamaModel) Name() string {
	return m.name
}

// GenerateContent calls the underlying model.
func (m *ollamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
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

// addHeaders sets the x-goog-api-client and user-agent headers
func (m *ollamaModel) addHeaders(headers http.Header) {
	headers.Set("x-goog-api-client", m.versionHeaderValue)
	headers.Set("user-agent", m.versionHeaderValue)
}

// modelName returns the model name to use for the API call.
// It prefers req.Model (which can be set by BeforeModelCallback),
// falling back to the construction-time name if unset.
func (m *ollamaModel) modelName(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}
	return m.name
}

// generate calls the model synchronously returning result from the first candidate.
func (m *ollamaModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	resp, err := m.client.Models.GenerateContent(ctx, m.modelName(req), req.Contents, req.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}
	if len(resp.Candidates) == 0 {
		// shouldn't happen?
		return nil, fmt.Errorf("empty response")
	}
	return converters.Genai2LLMResponse(resp), nil
}

// generateStream returns a stream of responses from the model.
func (m *ollamaModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()

	return func(yield func(*model.LLMResponse, error) bool) {
		for resp, err := range m.client.Models.GenerateContentStream(ctx, m.modelName(req), req.Contents, req.Config) {
			if err != nil {
				yield(nil, err)
				return
			}
			for llmResponse, err := range aggregator.ProcessResponse(ctx, resp) {
				if !yield(llmResponse, err) {
					return // Consumer stopped
				}
			}
		}
		if closeResult := aggregator.Close(); closeResult != nil {
			yield(closeResult, nil)
		}
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *ollamaModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

// mergeHeadersInterceptor is a http.RoundTripper that merges headers from the request
// with the model's headers before delegating to the base transport.
type mergeHeadersInterceptor struct {
	base http.RoundTripper
}

func (h *mergeHeadersInterceptor) RoundTrip(req *http.Request) (*http.Response, error) {
	for _, headerName := range []string{"x-goog-api-client", "user-agent"} {
		if values := req.Header.Values(headerName); len(values) > 0 {
			req.Header.Set(headerName, strings.Join(values, " "))
		}
	}

	if h.base == nil {
		return http.DefaultTransport.RoundTrip(req)
	}
	return h.base.RoundTrip(req)
}

func (m *ollamaModel) GetGoogleLLMVariant() genai.Backend {
	if m == nil || m.client == nil {
		return genai.BackendUnspecified
	}
	return m.client.ClientConfig().Backend
}

var _ googlellm.GoogleLLM = &ollamaModel{}
