package com.tars.chat;

/**
 * Represents a tool call from the LLM.
 */
public record ToolCall(String id, String name, String arguments) {}
