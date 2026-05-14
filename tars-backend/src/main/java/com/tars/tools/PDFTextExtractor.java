package com.tars.tools;

import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Utility class for extracting text from PDF documents.
 */
public class PDFTextExtractor {

    /**
     * Extracts text from a PDF document given its byte content.
     *
     * @param pdfBytes the PDF content as bytes
     * @return extracted text with page markers
     */
    public static String extractTextFromPdf(byte[] pdfBytes) {
        if (pdfBytes == null || pdfBytes.length == 0) {
            throw new IllegalArgumentException("PDF bytes must not be null or empty");
        }

        try (PDDocument document = PDDocument.load(pdfBytes)) {
            int pageCount = document.getNumberOfPages();
            StringBuilder extractedText = new StringBuilder();

            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(true);

            for (int i = 0; i < pageCount; i++) {
                extractedText.append("=== Page ").append(i + 1).append(" ===\n\n");

                // Fix: set page range so only the current page is extracted
                stripper.setStartPage(i + 1);
                stripper.setEndPage(i + 1);

                String pageText = stripper.getText(document);
                extractedText.append(pageText).append("\n\n");
            }

            return extractedText.toString();
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract text from PDF: " + e.getMessage(), e);
        }
    }

    /**
     * Extracts text from a PDF file given its path.
     *
     * @param filePath the path to the PDF file
     * @return extracted text with page markers
     */
    public static String extractTextFromPdfFile(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new IllegalArgumentException("File path must not be null or empty");
        }

        File file = new File(filePath);
        if (!file.exists()) {
            throw new IllegalArgumentException("File does not exist: " + filePath);
        }
        if (!file.isFile() || !file.canRead()) {
            throw new IllegalArgumentException("Path is not a readable file: " + filePath);
        }

        try (PDDocument document = PDDocument.load(file)) {
            int pageCount = document.getNumberOfPages();
            StringBuilder extractedText = new StringBuilder();

            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(true);

            for (int i = 0; i < pageCount; i++) {
                extractedText.append("=== Page ").append(i + 1).append(" ===\n\n");

                // Fix: set page range so only the current page is extracted
                stripper.setStartPage(i + 1);
                stripper.setEndPage(i + 1);

                String pageText = stripper.getText(document);
                extractedText.append(pageText).append("\n\n");
            }

            return extractedText.toString();
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract text from PDF: " + e.getMessage(), e);
        }
    }

    /**
     * Extracts text from a PDF document with markdown-like formatting.
     *
     * @param pdfBytes the PDF content as bytes
     * @return formatted text suitable for AI consumption
     */
    public static String extractTextAsMarkdown(byte[] pdfBytes) {
        if (pdfBytes == null || pdfBytes.length == 0) {
            throw new IllegalArgumentException("PDF bytes must not be null or empty");
        }

        try (PDDocument document = PDDocument.load(pdfBytes)) {
            int pageCount = document.getNumberOfPages();
            StringBuilder formattedText = new StringBuilder();

            formattedText.append("# PDF Document\n\n");
            formattedText.append("Total pages: ").append(pageCount).append("\n\n");
            formattedText.append("---\n\n");

            PDFTextStripper stripper = new PDFTextStripper();
            stripper.setSortByPosition(true);

            for (int i = 0; i < pageCount; i++) {
                formattedText.append("## Page ").append(i + 1).append("\n\n");

                // Fix: set page range so only the current page is extracted
                stripper.setStartPage(i + 1);
                stripper.setEndPage(i + 1);

                String pageText = stripper.getText(document);
                formattedText.append(formatParagraphs(pageText)).append("\n\n");
            }

            return formattedText.toString();
        } catch (IOException e) {
            throw new RuntimeException("Failed to extract text from PDF: " + e.getMessage(), e);
        }
    }

    /**
     * Formats text into paragraphs for better readability.
     */
    private static String formatParagraphs(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }

        // Split by double newlines to get paragraphs; filter blanks and rejoin
        return Arrays.stream(text.split("\n\\s*\n"))
                .map(String::trim)
                .filter(p -> !p.isEmpty())
                .collect(Collectors.joining("\n\n"));
    }
}