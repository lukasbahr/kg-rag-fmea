# Knowledge Graph Enhanced Retrieval-Augmented Generation for Failure Mode and Effects Analysisy

This repository contains the code, datasets, and additional resources for the research paper titled "Title of Your Research Paper," which is currently published as preprint in arXive. Here, we provide all the necessary information and resources to reproduce the results presented in our paper.

## Abstract

Failure mode and effects analysis (FMEA) is a critical tool for mitigating poten-
tial failures, particular during ramp-up phases of new products. However, its
effectiveness is often limited by the missing reasoning capabilities of the FMEA
tools, which are usually tabular structured. Meanwhile, large language models
(LLMs) offer novel prospects for fine-tuning on custom datasets for reasoning
within FMEA contexts. However, LLMs face challenges in tasks that require fac-
tual knowledge, a gap that retrieval-augmented generation (RAG) approaches
aim to fill. RAG retrieves information from a non-parametric data store and
uses a language model to generate responses. Building on this idea, we propose
to advance the non-parametric data store with a knowledge graph (KG). By
enhancing the RAG framework with a KG, our objective is to leverage analyt-
ical and semantic question-answering capabilities on FMEA data. This paper
contributes by presenting a new ontology for FMEA observations, an algorithm
for creating vector embeddings from the FMEA KG, and a KG enhanced RAG
framework. Our approach is validated through a human study and we measure
the performance of the context retrieval recall and precision.

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
├── data/                   # Example FMEA (csv)
├── .env_default            # Default environment variables
├── .gitignore              # The gitignore file
├── docker-compose.yml      # Neo4j Docker Compose file
├── requirements.txt        # The requirements file 
├── LICENSE                 # The license file
└── README.md               # The README file (this file)
```

## Setup and Installation

Instructions for setting up the environment and installing required dependencies.

```bash
# Example setup commands
git clone https://github.com/lukasbahr/kg-rag-fmea.git
cd kg-rag-fmea
pip install -r requirements.txt
```

## Usage

Instructions on how to start the backend service.

```bash
# Example usage commands
python code/kg_rag.py
```

## Additional Resources

- [Link to preprint](https://arxiv.org/abs/2406.18114)
- [Neo4j](https://neo4j.com/)

## Contributing

Please feel free to contact one of the authors in case you wish to contribute.

## License

This project is licensed under the [MIT License] - see the [MIT License](https://github.com/lukasbahr/kg-rag-fmea/blob/main/LICENSE) file for details.

## Contact Information

For any queries regarding the paper or the code, please open an issue on this repository or contact the authors directly at:

- [Lukas Bahr](mailto:lukas.bahr@bmw.de)