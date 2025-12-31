# README

This folder is intended for LLM-consumable documentation. The sample document is in [llmstxt](https://llmstxt.org/) format and seems to work well with chat clients. For instance, the following prompt worked well at generating good Lexos code with ChatGPT 4.1:

> Using the `kwic.txt` file as context, show me how to use Lexos to get KWIC result for the phrase "my soul" in Marlowe's play *Doctor Faustus*.

However, it did not use Lexos for loading or tokenising the document. For this, another prompt had to be made to rewrite the code using, for instance, the Lexos `tokenizer` module. Possibly this can be remedied by using a higher level file as context.

Further experimentation is needed.
