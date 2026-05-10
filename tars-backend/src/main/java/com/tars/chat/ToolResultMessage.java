package com.tars.chat;

/**
 * Tool result message containing the result of a tool execution.
 */
public final class ToolResultMessage extends ChatMessage {

    private final String toolCallId; // ← add this
    private final String toolName;
    private final String result;

    public ToolResultMessage(String toolCallId, String toolName, String result) {
        super("tool", result);
        this.toolCallId = toolCallId; // ← add this
        this.toolName = toolName;
        this.result = result;
    }

    public String toolCallId() { return toolCallId; } // ← add this

    public String toolName() { return toolName; }

    public String result() { return result; }

    public static ToolResultMessage from(ToolCall call, String result) {
        return new ToolResultMessage(call.id(), call.name(), result); // ← pass call.id()
    }

    @Override
    public String toString() {
        return "ToolResultMessage{" +
                "toolCallId='" + toolCallId + '\'' +
                ", toolName='" + toolName + '\'' +
                ", result='" + result + '\'' +
                '}';
    }
}
