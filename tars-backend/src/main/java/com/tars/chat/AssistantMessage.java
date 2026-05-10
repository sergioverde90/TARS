package com.tars.chat;

import java.util.List;

/**
 * Assistant message with optional text and/or tool calls.
 */
public final class AssistantMessage extends ChatMessage {
    private final String text;
    private final List<ToolCall> toolCalls;

    public AssistantMessage(String text, List<ToolCall> toolCalls) {
        super("assistant", text);
        this.text = text;
        this.toolCalls = toolCalls;
    }

    public String text() {
        return text;
    }

    public List<ToolCall> toolCalls() {
        return toolCalls;
    }

    public boolean hasToolCalls() {
        return toolCalls != null && !toolCalls.isEmpty();
    }

    public static AssistantMessage from(String text) {
        return new AssistantMessage(text, List.of());
    }

    public static AssistantMessage from(List<ToolCall> toolCalls) {
        return new AssistantMessage(null, toolCalls);
    }
}
