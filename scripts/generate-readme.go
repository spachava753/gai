package main

import (
	"fmt"
	"go/doc"
	"go/parser"
	"go/token"
	"log"
	"os"
	"regexp"
	"strings"
)

func main() {
	// Parse the current directory
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, ".", nil, parser.ParseComments)
	if err != nil {
		log.Fatalf("Failed to parse directory: %v", err)
	}

	// Find the gai package
	pkg, exists := pkgs["gai"]
	if !exists {
		log.Fatal("Could not find gai package")
	}

	// Create documentation from AST
	docPkg := doc.New(pkg, "./", 0)

	// Extract the package documentation
	packageDoc := docPkg.Doc
	if packageDoc == "" {
		log.Fatal("No package documentation found")
	}

	// Convert Go doc format to Markdown
	readme := convertDocToMarkdown(packageDoc)

	// Add badges and header
	header := `# gai - Go for AI

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Go Version](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)

`

	// Combine header with converted documentation
	fullReadme := header + readme

	// Write to README.md
	err = os.WriteFile("README.md", []byte(fullReadme), 0644)
	if err != nil {
		log.Fatalf("Failed to write README.md: %v", err)
	}

	fmt.Println("README.md generated successfully from doc.go!")
}

func convertDocToMarkdown(docText string) string {
	var result strings.Builder

	lines := strings.Split(docText, "\n")
	i := 0

	for i < len(lines) {
		line := lines[i]
		trimmedLine := strings.TrimRight(line, " \t")

		// Handle Go doc headers (# Header -> ## Header in markdown)
		if strings.HasPrefix(trimmedLine, "# ") {
			result.WriteString("## " + strings.TrimPrefix(trimmedLine, "# ") + "\n\n")
			i++
			continue
		}

		// Check if this line starts a code block
		if isCodeBlockStart(line) {
			// Collect all lines of the code block
			codeLines := []string{}
			for i < len(lines) && (isCodeBlockStart(lines[i]) || (strings.TrimSpace(lines[i]) == "" && i+1 < len(lines) && isCodeBlockStart(lines[i+1]))) {
				if isCodeBlockStart(lines[i]) {
					// Remove indentation
					cleanLine := removeIndentation(lines[i])
					codeLines = append(codeLines, cleanLine)
				} else {
					// Empty line within code block
					codeLines = append(codeLines, "")
				}
				i++
			}

			// Write the code block
			if len(codeLines) > 0 {
				lang := detectLanguage(codeLines)
				result.WriteString("```" + lang + "\n")
				for _, codeLine := range codeLines {
					result.WriteString(codeLine + "\n")
				}
				result.WriteString("```\n\n")
			}
			continue
		}

		// Handle regular text lines
		if trimmedLine == "" {
			result.WriteString("\n")
		} else {
			// Handle bullet points
			if isBulletPoint(trimmedLine) {
				result.WriteString(formatBulletPoint(trimmedLine) + "\n")
			} else {
				// Handle regular paragraph text
				result.WriteString(trimmedLine)
				// Add appropriate spacing
				if i+1 < len(lines) {
					nextLine := strings.TrimSpace(lines[i+1])
					if nextLine == "" || strings.HasPrefix(nextLine, "# ") || isBulletPoint(nextLine) || isCodeBlockStart(lines[i+1]) {
						result.WriteString("\n")
					} else {
						result.WriteString(" ")
					}
				} else {
					result.WriteString("\n")
				}
			}
		}
		i++
	}

	// Clean up the result
	output := result.String()

	// Fix multiple consecutive newlines
	re := regexp.MustCompile(`\n{3,}`)
	output = re.ReplaceAllString(output, "\n\n")

	// Ensure proper spacing around headers
	re = regexp.MustCompile(`([^\n])\n(## [^\n]+)`)
	output = re.ReplaceAllString(output, "$1\n\n$2")

	// Ensure proper spacing after headers
	re = regexp.MustCompile(`(## [^\n]+)\n([^\n])`)
	output = re.ReplaceAllString(output, "$1\n\n$2")

	return output
}

func isCodeBlockStart(line string) bool {
	// Must start with tab or 4+ spaces
	if !strings.HasPrefix(line, "\t") && !strings.HasPrefix(line, "    ") {
		return false
	}

	// Empty lines are not code block starts by themselves
	if strings.TrimSpace(line) == "" {
		return false
	}

	return true
}

func removeIndentation(line string) string {
	if strings.HasPrefix(line, "\t") {
		return strings.TrimPrefix(line, "\t")
	} else if strings.HasPrefix(line, "    ") {
		return strings.TrimPrefix(line, "    ")
	}
	return line
}

func isBulletPoint(line string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, "- ")
}

func formatBulletPoint(line string) string {
	trimmed := strings.TrimSpace(line)
	return trimmed
}

func detectLanguage(codeLines []string) string {
	// Join all lines to analyze content
	content := strings.Join(codeLines, "\n")

	// Check for Go-specific patterns
	goPatterns := []string{
		"package ",
		"func ",
		"type ",
		"import ",
		":=",
		"gai.",
		"context.",
		"fmt.",
		"err ",
		"var ",
		"const ",
		"interface{",
		"struct{",
	}

	for _, pattern := range goPatterns {
		if strings.Contains(content, pattern) {
			return "go"
		}
	}

	// Check for bash patterns
	bashPatterns := []string{
		"go get",
		"go install",
		"#!/bin/bash",
	}

	for _, pattern := range bashPatterns {
		if strings.Contains(content, pattern) {
			return "bash"
		}
	}

	// Default to no language
	return ""
}
