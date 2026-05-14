package com.tars.tools;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Base64;

/**
 * Tool for reading and extracting text from PDF documents.
 * Accepts base64-encoded PDF content and returns AI-readable text.
 */
@Component
public class PDFReadingTool {

    private static final Logger log = LoggerFactory.getLogger(PDFReadingTool.class);
    private static final ObjectMapper mapper = new ObjectMapper();
    private static final int MAX_CONTENT_LENGTH = 20000; // ~5000-8000 words

    public String readPdf(String pdfContent) {
        try {
            // Parse the input JSON to extract pdfContent
            JsonNode node = mapper.readTree(pdfContent);
            String base64Pdf = node.path("pdfContent").asText();

            if (base64Pdf == null || base64Pdf.isEmpty()) {
                return "Error: No PDF content provided. Please include 'pdfContent' as a base64-encoded string.";
            }

            // Decode base64 to bytes
            byte[] pdfBytes = Base64.getDecoder().decode(base64Pdf);

            // Check content size
            if (pdfBytes.length > MAX_CONTENT_LENGTH * 1024) { // Convert to bytes
                return "Error: PDF content exceeds maximum allowed size (" + MAX_CONTENT_LENGTH + " KB). Please upload a smaller document.";
            }

            // Extract text using PDFTextExtractor
            String extractedText = PDFTextExtractor.extractTextAsMarkdown(pdfBytes);

            // Add metadata and truncation notice if needed
            StringBuilder result = new StringBuilder();
            result.append(extractedText);

            if (extractedText.length() > MAX_CONTENT_LENGTH) {
                result.append("\n\n---\n\n");
                result.append("[Note: Content was truncated to ").append(MAX_CONTENT_LENGTH)
                        .append(" characters for token efficiency. The original PDF may contain more content.]");
            }

            return result.toString();

        } catch (Exception e) {
            log.error("PDFReadingTool error: {}", e.getMessage(), e);
            String errorMsg = "Error reading PDF: " + e.getMessage();

            // Provide more specific error messages for common issues
            if (e.getMessage() != null && e.getMessage().contains("stream")) {
                return "Error: Invalid or corrupted PDF file. Please ensure the file is a valid PDF format.";
            }
            if (e.getMessage() != null && e.getMessage().contains("decode")) {
                return "Error: Failed to decode PDF content. Please ensure the 'pdfContent' field contains valid base64-encoded data.";
            }

            return errorMsg;
        }
    }

}
