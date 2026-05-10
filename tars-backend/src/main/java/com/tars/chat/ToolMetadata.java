package com.tars.chat;

import java.util.Map;

/**
 * Metadata for a tool, including its name, description, and parameter schema.
 */
public record ToolMetadata(
    String name,
    String description,
    Map<String, Object> parameters  // JSON schema: {type, properties, required}
) {}
