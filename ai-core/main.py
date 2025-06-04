def main():
    print("Hello from python-ai-core!")


if __name__ == "__main__":
    main()




@app.post("/summarize")
def summarize(
    llm_inputs, 
    mode = 'overview'
): 
    pdf_content = PdfProcessor()

    summarizer()



