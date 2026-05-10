package com.tars.chat;

/**
 * System message from the developer.
 */
public final class SystemMessage extends ChatMessage {
    private final String text;

    public SystemMessage(String text) {
        super("system", text);
        this.text = text;
    }

    public String text() {
        return text;
    }

    public static SystemMessage from(String text) {
        return new SystemMessage(text);
    }
}
