"""All prompt templates for graph nodes.

Convention:
- Nodes using `llm.with_structured_output()` → prompt describes intent + field semantics only.
  Do NOT include a literal JSON block — the schema enforces the output format.
- Nodes returning free text (HyDE, Synthesizer) → prompt ends with a content cue.
- Nodes still using manual json.loads() (Global Summary) → prompt keeps full JSON template.

Note: Prompts are intentionally bilingual (English instructions, Vietnamese content cues)
because this system is designed to process Vietnamese documents and questions.
"""

# ============================================================
# Node 0: Context Compression
# [Output schema: ContextCompressionOutput]
# ============================================================
CONTEXT_COMPRESSION_PROMPT = """You are an expert Context Compressor. Your task is to summarize the provided document strictly for another AI agent to use as background context.

### OBJECTIVE:
Create a dense, highly concentrated summary that captures all essential entities, subjects, and core facts. This output will be used by downstream AI nodes to resolve pronoun references (like "it", "he", "this document") and plan sub-tasks. 

### STRICT RULES:
1. Maximum Information Density: Do not write for a human reader. Strip out all fluff, introductory phrases, and conversational text.
2. Entity Preservation (CRITICAL): You MUST extract and explicitly state all proper nouns, names of people, document titles, organizations, and key metrics. Downstream tasks will fail if these are omitted.
3. Objective State: Describe exactly what the text is about without assumptions or external knowledge.

### REQUIRED OUTPUT STRUCTURE:s
Return a dense and precise text block following this exact structure:
- MAIN TOPIC: [1 concise sentence defining the core subject]
- KEY ENTITIES: [Comma-separated list of specific names, items, or document references]
- CORE CONTEXT: [3-5 sentences strictly summarizing the most important facts and actions in the text]

### INPUT DOCUMENT:
{input_data}

### OUTPUT:
Generate the compressed context based on the rules above. Answer in the same language as the input document.
"""
# ============================================================
# Node 1: Ambiguity Checker
# [Output schema: AmbiguityCheckOutput]
# ============================================================
AMBIGUITY_CHECK_PROMPT = """
You are a semantic analysis expert. Your task is to determine whether the user's question contains enough information to perform a specific action.

### EVALUATION RULES:

1. EVALUATE AS CLEAR (is_ambiguous: false) WHEN:
   - The question is self-sufficient and meaningful on its own.
   - Or the question contains substitute pronouns/referents (it, this, author, this content, summarize for me...) AND there is corresponding information in the "Document summary" to map to.
   - Example: "Who is he?" when the Document summary is about "Steve Jobs".

2. EVALUATE AS AMBIGUOUS (is_ambiguous: true) WHEN:
   - The question uses substitute pronouns (it, they, that...) but the "Document summary" is empty or does not contain the corresponding subject.
   - The question consists of exclamations or single words without a clear purpose (Example: "Great", "Why?", "Really").
   - The question requests an action but the target cannot be identified (Example: "Fix it for me" - fix what?).

### INPUT DATA:
- User's question: {question}
- Current document summary: {content_summary} (Note: If this field is empty, it means the user has not provided any context/file).

### REJECTION REASON INSTRUCTION (CRITICAL):
If you evaluate the question as AMBIGUOUS (is_ambiguous: true), you MUST provide a `rejection_reason`. 
Explain exactly WHY the system cannot process the request and WHAT specific information the user needs to provide.
- Example 1: "Can you clarify what 'it' refers to in the document or context?"
- Example 2: "Your question is missing a subject; what specifically would you like me to fix or do?"
If the question is CLEAR (is_ambiguous: false), leave `rejection_reason` completely empty or null.

### REQUIRED OUTPUT:
Based on the defined Schema, please analyze and return the result.
"""
# ============================================================
# Node 2: Planner (Sub-task Decomposer)
# [Output schema: PlannerOutput]
# ============================================================
PLANNER_PROMPT = """You are an expert analyst and planner. Your task is to break down a complex question into smaller sub-tasks, each of which can be processed independently.

Rules:
1. If the question is simple, create only 1 sub-task (the original question itself).
2. If the question is complex or multi-part, split into 2-4 sub-tasks.
3. Each sub-task must be self-contained — understandable without reading the other tasks.
4. Do not over-split — each sub-task should have independent meaning.

Question: {question}

Return:
- sub_tasks: list of sub-tasks (at least 1 item)
- reasoning: brief explanation of how you decomposed the question"""

# ============================================================
# Node 3: Knowledge Router
# [Output schema: KnowledgeRouterOutput]
# ============================================================
KNOWLEDGE_ROUTER_PROMPT = """You are a knowledge router. You have access to a summary of the internal knowledge base (Global Summary) below.

Task: Determine whether the current task falls within the scope of the internal data.

=== GLOBAL SUMMARY (Internal Knowledge Base Index) ===
{global_summary}
=== END SUMMARY ===

Current task: {current_task}

Decision rules:
- If the task is directly related to topics, entities, or time periods in the Summary → choose "rag"
- If the task is completely outside the scope of the Summary (e.g. weather, current news, general knowledge) → choose "web"
- If the task is general knowledge that the LLM can answer directly → choose "llm_knowledge"

Return:
- route: one of "rag", "web", or "llm_knowledge"
- reasoning: brief explanation of why you chose this route"""

# ============================================================
# Node 4a: HyDE Generator
# [Free-text output — no structured output used]
# ============================================================
HYDE_PROMPT = """You are a document writing expert. Write a hypothetical document passage that answers the question/task below.

This passage does NOT need to be accurate — the goal is to produce content that is semantically similar to real documents in the database, so that when embedded, it will be close to the relevant chunks.

Requirements:
- Write 1-2 paragraphs (150-300 words)
- Use appropriate domain-specific terminology
- Write as if this is part of a real document

Task: {current_task}

Hypothetical document:"""

# ============================================================
# Node 6: Validator
# [Output schema: ValidatorOutput]
# ============================================================
VALIDATOR_PROMPT = """You are an information quality expert. Your task is to evaluate how complete the collected information is relative to the task requirements.

Task: {current_task}

### COLLECTED INFORMATION:
{all_context}

Evaluate based on:
1. What percentage of the question does the information answer?
2. What important information is still missing?
3. Is additional web search needed?

Return:
- completeness_score: score from 0.0 (completely missing) to 1.0 (fully complete)
- missing_info: description of missing information, or empty if sufficient
- needs_web_supplement: true if additional web search is needed
- web_search_query: web search query if needs_web_supplement=true, otherwise leave empty"""

# ============================================================
# Node 7: Synthesizer
# [Free-text output — no structured output used]
# ============================================================
SYNTHESIZER_PROMPT = """You are an information synthesis expert. Your task is to create a final answer based on ALL collected information.

Original question: {question}

=== COLLECTED INFORMATION ===
{all_context}
=== END ===

Rules:
1. Synthesize information from multiple sources coherently.
2. Prioritize information from the internal knowledge base (RAG) when available.
3. Supplement with web search information as needed.
4. Answer in the language of the question (Vietnamese or English).
5. If information is insufficient, clearly state which parts cannot be answered.
6. Cite sources where possible.

Answer:"""

# ============================================================
# Global Summary Builder
# [Still uses manual json.loads() → keep full JSON template]
# ============================================================
GLOBAL_SUMMARY_PROMPT = """You are a knowledge management expert. Create a Global Summary from the document passages below.

The summary must include:
1. **Main topics**: The major topics present in the data
2. **Key entities**: Names of people, organizations, products, projects, etc.
3. **Time range**: What time period does the data cover
4. **Content summary**: 2-3 sentences describing the overall content

=== DOCUMENT PASSAGES ===
{documents}
=== END ===

Reply in JSON format:
{{
    "topics": ["Topic 1", "Topic 2"],
    "entities": ["Entity 1", "Entity 2"],
    "time_range": "From ... to ...",
    "summary": "Overall content summary",
    "total_documents": number,
    "total_chunks": number
}}"""