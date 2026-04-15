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
// It uses Ollama's OpenAI-compatible API endpoint (/v1/chat/completions).
package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"

	"google.golang.org/genai"

	"github.com/rinaldes/adk-go/internal/llminternal"
	"github.com/rinaldes/adk-go/model"
)

type ollamaModel struct {
	client  *http.Client
	baseURL string
	name    string
}

// openAIChatRequest represents the request body for OpenAI chat completions API.
type openAIChatRequest struct {
	Model    string          `json:"model"`
	Messages []openAIMessage `json:"messages"`
	Tools    []openAITool    `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
}

type openAITool struct {
	Type     string             `json:"type"`
	Function openAIFunctionSpec `json:"function"`
}

type openAIFunctionSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   usage    `json:"usage"`
}

type choice struct {
	Index   int `json:"index"`
	Message struct {
		Role      string           `json:"role"`
		Content   string           `json:"content"`
		ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
	FinishReason string `json:"finish_reason"`
}

type streamChoice struct {
	Index int `json:"index"`
	Delta struct {
		Role      string           `json:"role,omitempty"`
		Content   string           `json:"content,omitempty"`
		ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
	} `json:"delta"`
	FinishReason string `json:"finish_reason"`
}

type openAIChatStreamResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// NewModel creates a new Ollama model client.
func NewModel(ctx context.Context, modelName string, _ *genai.ClientConfig) (model.LLM, error) {
	baseURL := ollamaBaseURL()
	return &ollamaModel{
		client:  &http.Client{},
		baseURL: baseURL,
		name:    modelName,
	}, nil
}

func ollamaBaseURL() string {
	if u := os.Getenv("OLLAMA_BASE_URL"); u != "" {
		return strings.TrimSuffix(u, "/")
	}
	return "http://localhost:11434"
}

func (m *ollamaModel) Name() string { return m.name }

func (m *ollamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)
	modelName := m.modelName(req)
	messages := convertContentsToMessages(req.Contents)
	tools := convertTools(req.Config)

	if stream {
		return m.generateStream(ctx, modelName, messages, tools)
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := m.generate(ctx, modelName, messages, tools)
		yield(resp, err)
	}
}

func (m *ollamaModel) generate(ctx context.Context, modelName string, messages []openAIMessage, tools []openAITool) (*model.LLMResponse, error) {
	chatReq := openAIChatRequest{
		Model:    modelName,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", m.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}

	var chatResp openAIChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from ollama")
	}

	return convertOpenAIResponseToLLMResponse(&chatResp), nil
}

func (m *ollamaModel) generateStream(ctx context.Context, modelName string, messages []openAIMessage, tools []openAITool) iter.Seq2[*model.LLMResponse, error] {
	aggregator := llminternal.NewStreamingResponseAggregator()
	return func(yield func(*model.LLMResponse, error) bool) {
		chatReq := openAIChatRequest{
			Model:    modelName,
			Messages: messages,
			Tools:    tools,
			Stream:   true,
		}

		body, err := json.Marshal(chatReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to marshal request: %w", err))
			return
		}

		url := fmt.Sprintf("%s/v1/chat/completions", m.baseURL)
		httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
		if err != nil {
			yield(nil, fmt.Errorf("failed to create request: %w", err))
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "text/event-stream")

		resp, err := m.client.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("failed to send request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			yield(nil, fmt.Errorf("ollama returned status %d", resp.StatusCode))
			return
		}

		scanner := bufio.NewScanner(resp.Body)

		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var streamResp openAIChatStreamResponse
			if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
				continue
			}

			if len(streamResp.Choices) == 0 {
				continue
			}

			choice := streamResp.Choices[0]
			genResp := openAIStreamToGenaiResponse(&streamResp, choice)

			for llmResponse, err := range aggregator.ProcessResponse(ctx, genResp) {
				if err != nil {
					yield(nil, err)
					return
				}
				if !yield(llmResponse, nil) {
					return
				}
			}

			if choice.FinishReason != "" {
				break
			}
		}

		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("stream error: %w", err))
			return
		}

		if finalResp := aggregator.Close(); finalResp != nil {
			yield(finalResp, nil)
		}
	}
}

func (m *ollamaModel) modelName(req *model.LLMRequest) string {
	if req.Model != "" {
		return req.Model
	}
	return m.name
}

func (m *ollamaModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}
	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

func convertContentsToMessages(contents []*genai.Content) []openAIMessage {
	messages := make([]openAIMessage, 0, len(contents))
	for _, content := range contents {
		if content == nil {
			continue
		}
		role := content.Role
		if role == "model" {
			role = "assistant"
		}

		var textParts []string
		for _, part := range content.Parts {
			if part.Text != "" {
				textParts = append(textParts, part.Text)
			}
		}
		if len(textParts) > 0 {
			messages = append(messages, openAIMessage{
				Role:    role,
				Content: strings.Join(textParts, "\n"),
			})
		}
	}
	return messages
}

func convertTools(cfg *genai.GenerateContentConfig) []openAITool {
	if cfg == nil || len(cfg.Tools) == 0 {
		return nil
	}

	var tools []openAITool
	for _, tool := range cfg.Tools {
		if tool == nil {
			continue
		}
		for _, decl := range tool.FunctionDeclarations {
			if decl == nil {
				continue
			}
			tools = append(tools, convertFunctionDeclaration(decl))
		}
	}
	return tools
}

func convertFunctionDeclaration(decl *genai.FunctionDeclaration) openAITool {
	spec := openAIFunctionSpec{
		Name:        decl.Name,
		Description: decl.Description,
	}

	if decl.Parameters != nil {
		schemaJSON, _ := json.Marshal(decl.Parameters)
		spec.Parameters = schemaJSON
	}

	return openAITool{
		Type:     "function",
		Function: spec,
	}
}

func openAIStreamToGenaiResponse(streamResp *openAIChatStreamResponse, choice streamChoice) *genai.GenerateContentResponse {
	content := &genai.Content{
		Role: genai.RoleModel,
	}
	if choice.Delta.Content != "" {
		content.Parts = []*genai.Part{{Text: choice.Delta.Content}}
	}

	finishReason := genai.FinishReasonUnspecified
	if choice.FinishReason != "" {
		finishReason = convertFinishReason(choice.FinishReason)
	}

	return &genai.GenerateContentResponse{
		Candidates: []*genai.Candidate{{
			Content:      content,
			FinishReason: finishReason,
		}},
		ModelVersion: streamResp.Model,
	}
}

func convertOpenAIResponseToLLMResponse(resp *openAIChatResponse) *model.LLMResponse {
	if len(resp.Choices) == 0 {
		return &model.LLMResponse{
			ErrorCode:    "EMPTY_RESPONSE",
			ErrorMessage: "No choices in response",
		}
	}

	choice := resp.Choices[0]
	content := &genai.Content{
		Role: "model",
	}

	if choice.Message.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{Text: choice.Message.Content})
	}

	for _, tc := range choice.Message.ToolCalls {
		if tc.Type == "function" {
			args := make(map[string]any)
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)

			content.Parts = append(content.Parts, &genai.Part{
				FunctionCall: &genai.FunctionCall{
					Name: tc.Function.Name,
					Args: args,
					ID:   tc.ID,
				},
			})
		}
	}

	finishReason := convertFinishReason(choice.FinishReason)

	return &model.LLMResponse{
		Content:      content,
		FinishReason: finishReason,
		ModelVersion: resp.Model,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		},
		TurnComplete: true,
	}
}

func convertFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonUnspecified
	}
}
