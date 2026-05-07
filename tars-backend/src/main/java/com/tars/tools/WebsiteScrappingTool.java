package com.tars.tools;

import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.springframework.stereotype.Component;

import java.net.InetAddress;
import java.net.URL;

@Component
public class WebsiteScrappingTool {

    @Tool("Fetches the content of a web page at the given URL and returns it as clean markdown text, including the page title, main content, and any relevant structured information. Use this when you need to read the actual content of a specific URL.")
    public String scrape(@P("the full URL to fetch, including https://") String url) {

        if (url == null || url.isEmpty()) {
            return "Error: URL is null or empty.";
        }

        try {
            // 1. Validate and Parse URL
            URL urls = new URL(url);

            // Security: Ensure we are not accessing internal networks (SSRF protection)
            String host = urls.getHost();
            InetAddress address = InetAddress.getByName(host);
            if (address.isLoopbackAddress() ||
                    address.isAnyLocalAddress() ||
                    address.isLinkLocalAddress() ||
                    address.isSiteLocalAddress()) {
                return "Error: Access to internal/local addresses is denied for security.";
            }

            // 2. Fetch and Parse HTML with Jsoup
            Document doc = Jsoup.connect(url)
                    .userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
                    .timeout(10000) // 10 seconds timeout
                    .followRedirects(true)
                    .get();

            // 3. Extract Key Information for AI Context

            // A. Title
            String title = doc.title();

            // B. Meta Description (AI often uses this for quick context)
            String description = "";
            Element metaDesc = doc.selectFirst("meta[name=description]");
            if (metaDesc != null) {
                description = metaDesc.attr("content").trim();
            }

            // C. Main Body Content (Stripping out nav, scripts, styles)
            String bodyContent = extractCleanBody(doc);

            // 4. Format for AI (Markdown-like structure)
            StringBuilder sb = new StringBuilder();
            sb.append("# ").append(title).append("\n\n");

            if (!description.isEmpty()) {
                sb.append("**Summary:** ").append(description).append("\n\n");
            }

            sb.append("---\n\n");
            sb.append(bodyContent);

            return sb.toString();

        } catch (Exception e) {
            return "Error scraping URL: " + e.getMessage();
        }
    }

    /**
     * Extracts clean text from the main body, removing scripts, styles, and nav.
     */
    private static String extractCleanBody(Document doc) {
        // Select the main content area.
        // We try 'article', 'main', 'content' first, fallback to 'body'.
        Element content = doc.selectFirst("article, main, [role=main], .content, body");

        if (content == null) {
            return "No main content found.";
        }

        // Remove unwanted elements (scripts, styles, nav, headers, footers)
        content.select("script, style, noscript, iframe, nav, footer, header, form, button, [onclick]").remove();

        // Get text and clean it up
        String text = content.text();

        // Clean up excessive whitespace (Jsoup.text() usually handles this, but we double-check)
        text = cleanWhitespace(text);

        // Optional: Limit length to prevent token overflow for LLMs
        // Return first 4000 characters (approx 1000-1500 words)
        if (text.length() > 4000) {
            text = text.substring(0, 4000) + "\n\n... [Content truncated for token efficiency]";
        }

        return text;
    }

    /**
     * Cleans up multiple newlines and spaces.
     */
    private static String cleanWhitespace(String text) {
        // Replace multiple newlines with a single newline
        return text.replaceAll("\\n\\s*\\n", "\n\n")
                .replaceAll("[ \\t]+", " ") // Replace tabs/multiple spaces with single space
                .trim();
    }

}