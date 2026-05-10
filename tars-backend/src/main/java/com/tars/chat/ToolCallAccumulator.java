package com.tars.chat;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Accumulates tool call data from SSE chunks.
 * Handles two formats:
 * 1. Streaming delta format: delta.tool_calls[{index, id, function: {name, arguments}}]
 * 2. Message format (mlx-openai-server): message.tool_calls[{id, type, function: {name, arguments}}]
 */
class ToolCallAccumulator {

    private final Map<String, ToolCallBuilder> builders = new HashMap<>();
    private final List<ToolCall> toolCalls = new ArrayList<>();

    private static class ToolCallBuilder {
        String id;
        String name;
        StringBuilder arguments = new StringBuilder();

        void addFunctionCall(String name, String args) {
            if (name != null) {
                this.name = this.name != null ? this.name + name : name;
            }
            if (args != null) {
                this.arguments.append(args);
            }
        }
    }

    // Handle streaming delta format: delta.tool_calls with index field
    public void processDeltaToolCalls(JsonNode toolCallsNode) {
        for (JsonNode tc : toolCallsNode) {
            JsonNode indexNode = tc.path("index");
            String key = indexNode.isInt() ? String.valueOf(indexNode.asInt()) : "0";

            ToolCallBuilder builder = builders.computeIfAbsent(key, k -> new ToolCallBuilder());

            JsonNode idNode = tc.path("id");
            if (idNode.isTextual()) {
                builder.id = idNode.asText();
            }

            JsonNode functionNode = tc.path("function");
            JsonNode nameNode = functionNode.path("name");
            JsonNode argsNode = functionNode.path("arguments");

            String name = nameNode.isTextual() ? nameNode.asText() : null;
            String args = argsNode.isTextual() ? argsNode.asText() : null;

            builder.addFunctionCall(name, args);
        }
    }

    // Handle message format: message.tool_calls with id, type, function structure
    public void processMessageToolCalls(JsonNode toolCallsNode) {
        for (JsonNode tc : toolCallsNode) {
            String key = tc.path("id").isTextual() ? tc.path("id").asText() : "0";

            ToolCallBuilder builder = builders.computeIfAbsent(key, k -> new ToolCallBuilder());

            JsonNode idNode = tc.path("id");
            if (idNode.isTextual()) {
                builder.id = idNode.asText();
            }

            JsonNode functionNode = tc.path("function");
            JsonNode nameNode = functionNode.path("name");
            JsonNode argsNode = functionNode.path("arguments");

            String name = nameNode.isTextual() ? nameNode.asText() : null;
            String args = argsNode.isTextual() ? argsNode.asText() : null;

            builder.addFunctionCall(name, args);
        }
    }

    public void markComplete() {
        if (!toolCalls.isEmpty()) return; // ✅ prevent double-flush
        for (ToolCallBuilder builder : builders.values()) {
            toolCalls.add(new ToolCall(builder.id, builder.name, builder.arguments.toString()));
        }
    }

    public List<ToolCall> getToolCalls() {
        return toolCalls;
    }

}
