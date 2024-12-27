# Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysis

This repository contains the code, datasets, and additional resources for the research paper titled "Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysis," which is currently published as a preprint on arXiv. Here, we provide all the necessary information and resources to reproduce the results presented in our paper.

## Abstract

Failure mode and effects analysis (FMEA) is an essential tool for mitigating potential failures, particularly during the ramp-up phases of new products. However, its effectiveness is often limited by the reasoning capabilities of the FMEA tools, which are usually tabular structured.
Meanwhile, large language models (LLMs) offer novel prospects for advanced natural language processing tasks. However, LLMs face challenges in tasks that require factual knowledge, a gap that retrieval-augmented generation (RAG) approaches aim to fill. RAG retrieves information from a non-parametric data store and uses a language model to generate responses.
Building on this concept, we propose to enhance the non-parametric data store with a knowledge graph (KG).
By integrating a KG into the RAG framework, we aim to leverage analytical and semantic question-answering capabilities for FMEA data.
This paper contributes by presenting set-theoretic standardization and a schema for FMEA data, a chunking algorithm for creating vector embeddings from the FMEA-KG, and a KG-enhanced RAG framework.
Our approach is validated through a user experience design study, and we measure the precision and performance of the context retrieval recall.

## Citation

If you find this work useful for your research, please cite our paper:

```
@misc{bahr2024knowledgegraphenhancedretrievalaugmented,
      title={Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysis}, 
      author={Lukas Bahr and Christoph Wehner and Judith Wewerka and José Bittencourt and Ute Schmid and Rüdiger Daub},
      year={2024},
      eprint={2406.18114},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2406.18114}, 
}
```

## Repository Structure

```
├── code/                   # All the code files used in the research
├── data/                   # Example FMEA (as csv)
├── .env_default            # Default environment variables
├── .gitignore              # The gitignore file
├── requirements.txt        # The requirements file 
├── LICENSE                 # The license file
└── README.md               # The README file (this file)
```

## Setup and Installation

Clone the repository and install the required packages:
1. Clone the repository and install the required Python packages
```bash
# Example setup commands
git clone https://github.com/lukasbahr/kg-rag-fmea.git
cd kg-rag-fmea
pip install -r requirements.txt
```
2. Ensure you have a Neo4j [instance](https://neo4j.com/product/neo4j-graph-database/) up and running. You can set it up locally or use a cloud solution.
3. Set up a GPT-4 instance. You can get access to GPT-4 via [OpenAI](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4-gpt-4-turbo-gpt-4o-and-gpt-4o-mini) or [Azure](https://azure.microsoft.com/de-de/products/ai-services/openai-service).
4. Configure the environment variables. Rename the .env_default file to .env and fill in the details for your Neo4j instance and GPT-4 API connection.

## Usage

Instructions on how to start the backend service.

```bash
# Example usage commands
python code/kg_rag.py
```

Example http request to the backend service.

```
@baseUrl = http://localhost:8080/api/v1

### Create FMEA graph from CSV path
POST {{baseUrl}}/create-fmea-graph
Content-Type: application/json

{
  "path": "example_fmea.csv"
}

### Run question answering
POST {{baseUrl}}/question-answer
Content-Type: application/json

{
  "question": "What failure mode has the highest RPN?"
}

### Set top_k for number of query results
POST {{baseUrl}}/set-top_k
Content-Type: application/json

{
  "top_k": 5
}
```

## Additional Resources

- [Link to preprint](https://arxiv.org/abs/2406.18114)
- [Neo4j](https://neo4j.com/)
- [OpenAI](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4-gpt-4-turbo-gpt-4o-and-gpt-4o-mini)
- [Azure](https://azure.microsoft.com/de-de/products/ai-services/openai-service)

## Contributing

Please feel free to contact one of the authors in case you wish to contribute.

## License

This project is licensed under the MIT License - see the [MIT License](https://github.com/lukasbahr/kg-rag-fmea/blob/main/LICENSE) file for details.

## Contact Information

For any queries regarding the paper or the code, please open an issue on this repository or contact the authors directly at:

- [Lukas Bahr](mailto:lukas.bahr@bmw.de)