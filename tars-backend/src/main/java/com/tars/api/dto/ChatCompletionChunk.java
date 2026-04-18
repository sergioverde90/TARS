package com.tars.api.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record ChatCompletionChunk(
    String id,
    String object,
    long created,
    String model,
    List<Choice> choices
) {
    public record Choice(
        int index,
        Delta delta,
        @JsonProperty("finish_reason") String finishReason
    ) {}

    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Delta(
        String content,
        @JsonProperty("reasoning_content") String reasoningContent
    ) {}

    public static ChatCompletionChunk contentDelta(String content) {
        return chunk(new Delta(content, null), null);
    }

    public static ChatCompletionChunk finish() {
        return chunk(new Delta(null, null), "stop");
    }

    private static ChatCompletionChunk chunk(Delta delta, String finishReason) {
        return new ChatCompletionChunk(
            "chatcmpl-tars",
            "chat.completion.chunk",
            System.currentTimeMillis() / 1000,
            "local",
            List.of(new Choice(0, delta, finishReason))
        );
    }
}
