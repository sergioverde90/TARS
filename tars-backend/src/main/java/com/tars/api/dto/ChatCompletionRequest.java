package com.tars.api.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public record ChatCompletionRequest(
    String model,
    List<Message> messages,
    Boolean stream,
    List<Tool> tools,
    @JsonProperty("tool_choice") Object toolChoice,
    Double temperature,
    Double topP,
    @JsonProperty("n") Integer n,
    @JsonProperty("max_tokens") Integer maxTokens,
    @JsonProperty("max_completion_tokens") Integer maxCompletionTokens,
    List<String> stop,
    @JsonProperty("frequency_penalty") Double frequencyPenalty,
    @JsonProperty("presence_penalty") Double presencePenalty,
    ResponseFormat responseFormat,
    Long seed,
    String user,
    @JsonProperty("min_p") Double minP,
    @JsonProperty("repeat_penalty") Double repeatPenalty,
    @JsonProperty("thinking_budget_tokens") Integer thinkingBudgetTokens,
    @JsonProperty("chat_template_kwargs") Map<String, String> chatTemplateKwargs
) {
    public record Message(String role, List<ContentItem> content) {}

    /**
     * A content item within a message.
     * <p>
     * Supported types:
     * <ul>
     *   <li>{@code "text"} — plain text content (e.g. user prompt, assistant reply)</li>
     *   <li>{@code "pdf"} — base64-encoded PDF file content</li>
     * </ul>
     */
    public record ContentItem(@JsonProperty("type") String type, @JsonProperty("text") String text) {}

    public record Tool(@JsonProperty("type") String type, FunctionSpec function) {
        public record FunctionSpec(
            @JsonProperty("name") String name,
            @JsonProperty("description") String description,
            @JsonProperty("parameters") Object parameters
        ) {}
    }

    public record ResponseFormat(String type) {}

    public boolean isStream() {
        return stream != null && stream;
    }
}
