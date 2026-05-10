package com.tars.chat;

/**
 * Base class for chat messages.
 */
public abstract class ChatMessage {
    private final String role;
    private final String content;

    protected ChatMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    public String role() {
        return role;
    }

    public String content() {
        return content;
    }
}
