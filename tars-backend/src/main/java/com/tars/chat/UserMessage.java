package com.tars.chat;

/**
 * User message with text content.
 */
public final class UserMessage extends ChatMessage {
    private final String text;

    public UserMessage(String text) {
        super("user", text);
        this.text = text;
    }

    public String text() {
        return text;
    }

    public static UserMessage from(String text) {
        return new UserMessage(text);
    }
}
