# AI Shield: Advanced Security Layer for Safe and Reliable AI Interactions

1. Introduction  
   1.1 Importance of Security and Reliability in AI Model Interactions  
   1.2 Objectives of the Script  
2. Technical Operation of the Script  
   2.1 Configuration and Initialization  
   2.2 Input Scanners  
   2.3 Output Scanners  
   2.4 Initialization of FastAPI Application and Pydantic Model  
   2.5 API Interaction  
   2.6 Endpoint for Handling Chat Completion Requests  
3. Functional Examples  
4. Conclusions  
5. Libraries Used and Additional References  
6. Acknowledgments  

# 1. Introduction  
1.1 Importance of Security and Reliability in AI Model Interactions  
In corporate settings, the security and reliability of interactions with AI models (such as Large Language Models, LLMs) are crucial for several reasons:  
- **Protection of Sensitive Data**: AI models may process sensitive or confidential information. Ensuring that data is not compromised or inadvertently exposed through inappropriate responses is essential.  
- **Avoiding Toxic Content**: AI models may generate responses containing toxic or offensive content if not properly filtered. This could harm the company's reputation and undermine customer trust.  
- **Improving Reliability**: Ensuring that responses are relevant and accurate is fundamental to maintaining service quality and meeting customer expectations.  
- **Prevention of Attacks and Manipulations**: AI models can be vulnerable to manipulations such as prompt injection or data poisoning attempts. Implementing security measures helps prevent such attacks and ensures system integrity.  

> *Prompt Injection*: A technique used by malicious actors to disguise harmful inputs as legitimate prompts to manipulate generative AI systems into revealing sensitive information.

1.2 Objectives of the Script  
The primary goal of the script is to mitigate risks associated with prompts and responses in the following ways:  
- **Prompt Filtering**: Input scanners analyze prompts to ensure they do not contain toxic, inappropriate, or potentially harmful content, preventing the model from generating responses based on problematic inputs.  
- **Response Validation**: Output scanners verify that generated responses are appropriate and relevant, avoiding sensitive content or responses that do not meet company standards.  
- **Security and Compliance**: The script ensures that only secure and compliant data is transmitted to the AI model, protecting the company from reputational and legal risks.  

To achieve these objectives, *LLM Guard* was developed as a system designed to monitor and manage the use of large language models (LLMs). Its primary purpose is to ensure the secure, ethical, and compliant use of these models.  

# 2. Technical Operation of the Script  
2.1 Configuration and Initialization  
The script reads configurations from a `config.ini` file, setting key parameters such as the API URL, AI model, and authentication token. Additionally, a "Vault" is created to securely manage sensitive data.

2.2 Input Scanners  
A list of input scanners is defined to analyze and sanitize user-submitted prompts. The input scanners implemented in the script are:  
- **Anonymize**: Removes or obscures personal or sensitive information in the prompt, safeguarding user privacy and preventing accidental transmission of sensitive data to the AI model.  
- **Toxicity**: Detects and filters toxic or offensive content within the prompt, preventing the use of inappropriate or potentially harmful language in generating responses.  
- **TokenLimit**: Ensures the prompt does not exceed a maximum number of tokens or words, allowing the AI model to handle the input efficiently without performance issues or errors.  
- **PromptInjection**: Identifies and neutralizes prompt injection attempts.

2.3 Output Scanners  
A list of output scanners is used to analyze and sanitize the AI-generated responses before they are returned to the user. The output scanners implemented in the script are:  
- **Deanonymize**: Reintegrates previously obscured personal information only when it is necessary and safe to do so.  
- **NoRefusal**: Ensures the model provides useful and relevant responses rather than refusing to address the user's question. 
- **Relevance**: Verifies that the generated response is relevant to the provided prompt, avoiding off-topic or unrelated answers.  
- **Sensitive**: Detects potentially sensitive or inappropriate content in the AI-generated response. If inappropriate content is found, the scanner can modify or flag it.

2.4 Initialization of FastAPI Application and Pydantic Model  
The script initializes a FastAPI application, a high-performance Python web framework. FastAPI facilitates the creation of RESTful APIs and efficiently handles HTTP requests.  
A Pydantic model is defined to describe the structure of the data expected by the API as input.

2.5 API Interaction  
The asynchronous function `get_response` enables the user's prompt to be sent to an external AI API (e.g., `https://api.regolo.ai/v1/chat/completions`) and retrieves the response. In this function, the user's prompt is encapsulated in a JSON payload, along with the specified model (e.g., `mistralai/Mistral7B-Instruct-v0.2`).

2.6 Endpoint for Handling Chat Completion Requests  
This endpoint manages POST requests sent to the `/v1/chat/completions` URL.   
- **Prompt Sanitization**: Upon arrival, the user's prompt is analyzed and sanitized using the input scanners. If the prompt is invalid, the endpoint returns an HTTP 400 error.  
- **Request to External API**: The sanitized prompt is sent to the external API using the `get_response` function, and the AI-generated response is retrieved.  
- **Response Sanitization**: The API's response is also analyzed and sanitized using the output scanners. If the response is invalid, an HTTP 400 error is returned.  
- **Final Response**: If both the prompt and the response pass all checks, the endpoint returns the sanitized prompt and response in JSON format, allowing users to observe how information was securely obscured and reintroduced effectively.  

This process ensures that AI interactions are safe and compliant with corporate standards.

# 3. Functional Examples  
- In the first example, the prompt containing sensitive information about the client "Mario Bianchi" is appropriately sanitized. In the final response, the data is reintroduced securely and effectively.  
- In the second example, the effectiveness of the "Toxicity" input scanner is demonstrated. It detects and filters the toxic content in the prompt (e.g., constructing a bomb), preventing the use of potentially harmful language in generating responses.
(for the images take a look at the report file in the repository)

# 4. Conclusions  
This script helps maintain high standards of security and reliability in interactions with AI models, protecting companies from risks and ensuring a high-quality user experience. Additional input and output scanners can be implemented to further enhance the system's security, ethics, and regulatory compliance.

# 5. Libraries Used and Additional References  
- [LLM Guard](https://llm-guard.com/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- [Regolo AI](https://regolo.ai/)

# 6. Acknowledgments  
Special thanks to Marco Cristofanilli for guidance during the development of this script and for the opportunity to gain expertise in these advanced AI technologies.  
