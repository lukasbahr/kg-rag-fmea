# -*- coding: utf-8 -*

from os import getenv, environ
import re
import collections
import openai
from dotenv import load_dotenv
import json
import connexion
import pandas as pd
from langchain.vectorstores import Neo4jVector
from langchain.graphs import Neo4jGraph
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()
API_KEY = getenv("AZURE_OPENAI_API_KEY")
RESOURCE_ENDPOINT = getenv("AZURE_OPENAI_API_BASE")
NEO4J_URL = getenv("NEO4J_URL")
NEO4J_USERNAME = getenv("NEO4J_USER")
NEO4J_PASSWORD = getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = getenv("NEO4J_DATABASE")

environ["OPENAI_API_TYPE"] = "azure"
environ["OPENAI_API_BASE"] = RESOURCE_ENDPOINT
environ["OPENAI_API_KEY"] = API_KEY
environ["OPENAI_API_VERSION"] = "2022-12-01"

openai.api_type = "azure"
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-07-01-preview"
openai.api_key = API_KEY

# CYPER QUERIES
MERGE_NODE_QUERY = "MERGE ({nodeRef}:{node} {properties})"

MERGE_RELATION_QUERY = "MERGE ({nodeRef1})-[:{relation}]->({nodeRef2})"

MATCH_QUERY = "MATCH ({nodeRef}:{node} {properties})"

TRAVERSE_QUERY = """
MATCH (fm:FailureMeasure)<-[:isImprovedByFailureMeasure]-(fc:FailureCause)<-[:isDueToFailureCause]-(fe:FailureEffect)-[:occursAtProcessStep]->(ps:ProcessStep)
WITH fm, fc, fe, ps
MATCH (fe)-[:resultsInFailureConsequence]->(fco:FailureConsequence)
WHERE ID(fe)={id}
RETURN fm, fc, fe, fco, ps, ID(fm), ID(fc), ID(fe), ID(fco), ID(ps);
"""

# TEMPLATES INFERENCES JOBS
CYPHER_GENERATION_TEMPLATE = """
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
If the question contains a relationship type that is not provided by the schema, but is 
similiar to relationship types from the schema, choose the most similar one instead. 
Task:
Generate Cypher statement to query a graph database.
Schema:
{schema}
Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Always return a Cypher statement, even if you don't know the answer.
"""

CYPHER_QUESTION_TEMPLATE = """
Task:
Generate Cypher statement to query a graph database.
Question:
{question}
"""

CYPHER_QA_TEMPLATE = """
Task:
As an assistant, your task is to provide helpful and human understandable answers based on the provided context (JSON datastructure). 
The context given is authoritative, and you must never doubt it or try to correct it. 
Your answer should sound like a natural response to the question, and you should not mention that you based the result on the given context. 
If the provided context is empty, you should state that you don't know the answer. 
Context:
"{context}"
Question:
{question} 
"""

ANSWER_SUMMARIZE_TEMPLATE = """
Task:
As an assistant, your task is to summarize the information such that it answers the question and can be processed in a further inference job.
Information:
"{information}"
Question:
"{question}"
"""


class Neo4JRepository(Neo4jVector, Neo4jGraph):
    """Neo4J Repository."""

    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        database: str,
        embedding: OpenAIEmbeddings,
    ) -> None:
        super().__init__(
            url=url,
            username=username,
            password=password,
            database=database,
            embedding=embedding,
        )

        self.refresh_schema()


class KGRAGService(Neo4JRepository):
    """KG RAG Service for FMEA."""

    def __init__(self):
        super().__init__(
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
            embedding=OpenAIEmbeddings(),
        )

        self.top_k = 3

        self.context_cypher = [
            dict(
                role="system",
                content=CYPHER_GENERATION_TEMPLATE.format(schema=self.schema),
            )
        ]
        self.context_qa = collections.deque(maxlen=1)

    @staticmethod
    def extract_cypher(text: str) -> str:
        """
        Extracts cypher from a string containing LLM output.

        Args:
            text (str): A string containing LLM output.

        Returns:
            str: The input string with quotes removed.
        """
        pattern = r"```(.*?)```"

        matches = re.findall(pattern, text, re.DOTALL)

        return matches[0] if matches else text.replace("\n", " ")

    @staticmethod
    def format_properties(properties: dict) -> str:
        """
        Formats a dictionary of properties into a string representation.

        Args:
            properties (dict): A dictionary of properties to format.

        Returns:
            str: A string representation of the formatted properties.
        """
        properties_str: str = "{"

        for key, value in properties.items():
            properties_str += f'{key}: "{value}",'
            if key == "A" or key == "B" or key == "E" or key == "RPZ":
                properties_str += f"{key}: {value},"

        properties_str = properties_str.strip(",") + "}"

        return properties_str

    def qa_prompt_context(self, question: str, context: str) -> None:
        """
        Adds a question and context to the QA context.

        Args:
            question (str): The question to be added.
            context (str): The context to be added.

        Returns:
            None
        """
        prompt = CYPHER_QA_TEMPLATE.format(
            context=context,
            question=question,
        )
        self.context_qa.append(dict(role="assistant", content=prompt))

    def cypher_prompt_context(self, question: str) -> None:
        """
        Adds a question to the Cypher context.

        Args:
            question (str): The question to be added.

        Returns:
            None
        """
        prompt = CYPHER_QUESTION_TEMPLATE.format(
            question=question,
        )
        if len(self.context_cypher) == 1:
            self.context_cypher.append(dict(role="system", content=prompt))
        else:
            self.context_cypher[1] = dict(role="system", content=prompt)

    def summarize_context(self, context: str, question: str):
        """
        Summarize the context.

        Args:
            context (str): The context to summarize.
            question (str): The question to summarize.

        Returns:
            dict: The summarized context.
        """
        prompt = ANSWER_SUMMARIZE_TEMPLATE.format(
            information=context, question=question
        )
        return dict(role="assistant", content=prompt)

    def set_top_k(self, top_k: int):
        """
        Set the top k value.

        Args:
            top_k (int): The top k value to set.

        Returns:
            true
        """
        self.top_k = top_k
        return True

    def run_inference(
        self, context: [dict], temperature: float = 0.0, max_tokens: int = 4000
    ) -> str:
        """
        Run inference on the OpenAI API.

        Args:
            context (list): A list of dictionaries containing the context.
            temperature (float): The temperature to use for the inference.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text.
        """
        return openai.ChatCompletion.create(
            engine="gpt-4",
            messages=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def traverse_graph(self, failureEffectId: str) -> list[dict]:
        """
        Returns a list of nodes and relations for a given failure measure id.

        Args:
            failureMeasureId (str): The failure measure id to traverse the graph for.

        Returns:
            list[dict]: A list of nodes and relations.
        """
        try:
            result = self.query(TRAVERSE_QUERY.format(id=failureEffectId))
            return result
        except Exception as e:
            print(e)

    def validate_cypher(self, cypher: str) -> bool:
        """
        Validate a Cypher query.

        Args:
            cypher (str): The Cypher query to validate.

        Returns:
            bool: True if the Cypher query is valid, False otherwise.
        """
        try:
            self.query(cypher)
            return True
        except Exception:
            return False

    def get_failure_effect_ids(self) -> list[dict]:
        """
        Get all failure effect ids.

        Returns:
            list[dict]: A list of failure effect ids.
        """
        try:
            result = self.query(
                """
                    MATCH (fe:FailureEffect)
                    RETURN ID(fe);
                    """
            )
            return result
        except Exception as e:
            print(e)

    def get_failure_measure_ids(self) -> list[dict]:
        """
        Get all failure measure ids.

        Returns:
            list[dict]: A list of failure measure ids.
        """
        try:
            result = self.query(
                """
                    MATCH (fm:FailureMeasure)
                    RETURN ID(fm);
                    """
            )
            return result
        except Exception as e:
            print(e)

    def create_fmea_graph(self, csv_file: str) -> bool:
        """
        Create the FMEA graph.

        Args:
            csv_file (str): The path to the csv file containing the FMEA data.

        Returns:
            bool: True if the graph was created successfully, False otherwise.

        """
        df = pd.read_csv(csv_file, delimiter=";", encoding="utf-8")

        # Create nodes and relations
        for _, row in df.iterrows():
            nodes = [
                MERGE_NODE_QUERY.format(
                    nodeRef="ProcessStep",
                    node="ProcessStep",
                    properties=self.format_properties(
                        {"Prozessschritt": row["ProcessStep"]}
                    ),
                ),
                MERGE_NODE_QUERY.format(
                    nodeRef="FailureConsequence",
                    node="FailureConsequence",
                    properties=self.format_properties(
                        {
                            "Fehlerfolge": row["FailureConsequence"],
                            "B": row["B"],
                        }
                    ),
                ),
                MERGE_NODE_QUERY.format(
                    nodeRef="FailureEffect",
                    node="FailureEffect",
                    properties=self.format_properties(
                        {
                            "Fehlerart": row["FailureEffect"],
                            "RPZ": row["RPZ"],
                        }
                    ),
                ),
                MERGE_NODE_QUERY.format(
                    nodeRef="FailureCause",
                    node="FailureCause",
                    properties=self.format_properties(
                        {
                            "Fehlerursache": row["FailureCause"],
                            "A": row["A"],
                        }
                    ),
                ),
                MERGE_NODE_QUERY.format(
                    nodeRef="FailureMeasure",
                    node="FailureMeasure",
                    properties=self.format_properties(
                        {
                            "Vermeidungsmassnahme": row["FailureMeasure"],
                            "Entdeckungsmassnahme": row["DetectionMeasure"],
                            "E": row["E"],
                        }
                    ),
                ),
            ]

            relations = [
                MERGE_RELATION_QUERY.format(
                    nodeRef1="FailureEffect",
                    relation="occursAtProcessStep",
                    nodeRef2="ProcessStep",
                ),
                MERGE_RELATION_QUERY.format(
                    nodeRef1="FailureEffect",
                    relation="resultsInFailureConsequence",
                    nodeRef2="FailureConsequence",
                ),
                MERGE_RELATION_QUERY.format(
                    nodeRef1="FailureEffect",
                    relation="isDueToFailureCause",
                    nodeRef2="FailureCause",
                ),
                MERGE_RELATION_QUERY.format(
                    nodeRef1="FailureCause",
                    relation="isImprovedByFailureMeasure",
                    nodeRef2="FailureMeasure",
                ),
            ]

            query = " \n ".join(nodes + relations)

            try:
                self.query(query)
            except Exception:
                return False

        # Create vector embeddings
        self.create_vector_embeddings()

        return True

    def create_vector_embeddings(self) -> bool:
        """
        Create vector embeddings for the FMEA graph.

        Returns:
            bool: True if the vector embeddings were created successfully, False otherwise.
        """
        # Get all failure measure ids
        # failureMeasureIds = self.get_failure_measure_ids()

        failureEffectIds = self.get_failure_effect_ids()

        # Check if the index already exists
        embedding_dimension = self.retrieve_existing_index()

        # If the index doesn't exist
        if not embedding_dimension:
            self.create_new_index()

        # Add the failure measures to the index
        for entry in failureEffectIds:
            id = entry["ID(fe)"]
            nodes = self.traverse_graph(str(id))
            chunk, nodeIds = self.create_chunk(nodes)

            embeddedNodeId = self.add_texts([chunk], metadatas=[nodeIds])[0]

            query = [
                MATCH_QUERY.format(
                    nodeRef="index",
                    node="Chunk",
                    properties=self.format_properties({"id": embeddedNodeId}),
                ),
                "WITH index ",
                MATCH_QUERY.format(
                    nodeRef="fe",
                    node="FailureEffect",
                    properties=self.format_properties({}),
                ),
                "WHERE ID(fe)={id}".format(id=id),
                MERGE_RELATION_QUERY.format(
                    nodeRef1="fe",
                    relation="isIndexed",
                    nodeRef2="index",
                ),
            ]

            try:
                self.query("\n".join(query))
            except Exception as e:
                raise e

        return True

    def create_chunk(self, nodes: [dict]) -> str:
        """
        Create a chunk from a list of nodes.

        Args:
            nodes (list[dict]): A list of nodes.

        Returns:
            str: The chunk.
        """
        fm, fc, fe, fco, ps = [[] for _ in range(5)]

        nodeIds = {
            "failureMeasureIds": [],
            "failureCauseIds": [],
            "failureEffectIds": [],
            "failureConsequenceIds": [],
            "processStepIds": [],
        }

        for node in nodes:
            if node["fm"] not in fm:
                fm.append(node["fm"])
                nodeIds["failureMeasureIds"].append(node["ID(fm)"])
            if node["fc"] not in fc:
                fc.append(node["fc"])
                nodeIds["failureCauseIds"].append(node["ID(fc)"])
            if node["fe"] not in fe:
                fe.append(node["fe"])
                nodeIds["failureEffectIds"].append(node["ID(fe)"])
            if node["fco"] not in fco:
                fco.append(node["fco"])
                nodeIds["failureConsequenceIds"].append(node["ID(fco)"])
            if node["ps"] not in ps:
                ps.append(node["ps"])
                nodeIds["processStepIds"].append(node["ID(ps)"])

        chunk = (
            ", ".join("Prozessschritt: " + i["Prozessschritt"] for i in ps)
            + "".join(
                ", Fehlerfolge: " + i["Fehlerfolge"] + ", B: " + str(i["B"])
                for i in fco
            )
            + "".join(
                ", Fehlerart: " + i["Fehlerart"] + ", RPZ: " + str(i["RPZ"]) for i in fe
            )
            + "".join(
                ", Fehlerursache: " + i["Fehlerursache"] + ", A: " + str(i["A"])
                for i in fc
            )
            + "".join(
                ", Vermeidungsmassnahme: "
                + i["Vermeidungsmassnahme"]
                + ", Entdeckungsmassnahme: "
                + i["Entdeckungsmassnahme"]
                + ", E: "
                + str(i["E"])
                for i in fm
            )
        )

        return chunk, nodeIds

    def answer_question(self, question: str) -> dict:
        """
        Run answer question RAG service.

        Args:
            question (str): The question to answer.

        Returns:
            dict: The answer and context.
        """
        # List pre answers
        pre_answer = list()

        # Add question to cypher context
        self.cypher_prompt_context(question)

        # Run inference
        result = self.run_inference(self.context_cypher)

        # Extract cypher query
        cypher_query = self.extract_cypher(result.choices[0].message.content)

        # Check if cypher query is valid
        if self.validate_cypher(cypher_query):
            query_result = self.query(cypher_query)
            if len(query_result) > self.top_k:
                query_result = query_result[: self.top_k]
        else:
            query_result = None

        if not query_result:
            # Vector search
            results = self.similarity_search(question, k=self.top_k)
            query_result = [result.page_content for result in results]

        # Summarize the query results for further processing
        for result in query_result:
            result_summarize = self.run_inference(
                [self.summarize_context(context=json.dumps(result), question=question)]
            )
            pre_answer.append(result_summarize.choices[0].message.content)

        # Add question and context to QA context
        self.qa_prompt_context(question, json.dumps(pre_answer))

        # Run inference
        answer = self.run_inference(list(self.context_qa), temperature=1.0)

        return {
            "answer": answer.choices[0].message.content,
            "context": pre_answer,
            "context_raw": query_result,
        }


# RAG SERVICE
rag_service = KGRAGService()


# API ENDPOINTS
def create_graph(body: object):
    return rag_service.create_fmea_graph(csv_file=body["path"])


def answer_question(body: object):
    return rag_service.answer_question(body["question"])


def set_top_k(body: object):
    return rag_service.set_top_k(body["top_k"])


# MAIN ENTRYPOINT
if __name__ == "__main__":
    app = connexion.FlaskApp(__name__)
    app.add_api("api.yml")
    application = app.app
    app.run(port=8080)
