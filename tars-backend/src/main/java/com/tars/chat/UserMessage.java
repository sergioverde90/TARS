package com.tars.chat;

import java.util.Map;

/**
 * User message with text content.
 */
public final class UserMessage extends ChatMessage {

    private final String text;
    private final Map<String, String> meta;

    private UserMessage(String text, Map<String, String> meta) {
        super("user", text);
        this.text = text;
        this.meta = meta;
    }

    public String text() {
        return text;
    }

    public static UserMessage from(String text, Map<String, String> meta) {
        return new UserMessage(text, meta);
    }
}
