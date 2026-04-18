package com.tars.api.dto;

import java.util.List;

public record ChatCompletionRequest(
    String model,
    List<Message> messages,
    Boolean stream
) {
    public record Message(String role, String content) {}

    public boolean isStream() {
        return stream != null && stream;
    }
}
