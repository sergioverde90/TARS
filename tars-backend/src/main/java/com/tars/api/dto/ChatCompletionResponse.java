package com.tars.api.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record ChatCompletionResponse(
    String id,
    String object,
    long created,
    String model,
    List<Choice> choices
) {
    public record Choice(
        int index,
        Message message,
        @JsonProperty("finish_reason") String finishReason
    ) {}

    public record Message(String role, String content) {}

    public static ChatCompletionResponse of(String content) {
        return new ChatCompletionResponse(
            "chatcmpl-tars",
            "chat.completion",
            System.currentTimeMillis() / 1000,
            "local",
            List.of(new Choice(0, new Message("assistant", content), "stop"))
        );
    }
}
