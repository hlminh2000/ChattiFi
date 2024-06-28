import { DynamicStructuredTool } from "@langchain/core/tools"
import { createToolCallingAgent, AgentExecutor } from "langchain/agents"
import { ChatPromptTemplate } from "@langchain/core/prompts"
import { ChatGroq } from "@langchain/groq"
import { BufferMemory, CombinedMemory } from "langchain/memory"
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai"
import { MemoryVectorStore } from "langchain/vectorstores/memory"
import { ConsoleCallbackHandler } from "@langchain/core/tracers/console"
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { PromptTemplate } from '@langchain/core/prompts'

import * as dotenv from "@dotenvx/dotenvx"
import _ from 'lodash'
import z from 'zod'
import inquirer from "inquirer"


(async () => {

  dotenv.config()
  const openAIApiKey = process.env.OPENAI_API_KEY
  const groqApiKey = process.env.GROQ_API_KEY
  const fmpApiKey = process.env.FMP_API_KEY
  if (!(openAIApiKey || groqApiKey)) 
    throw "Either OPENAI_API_KEY or GROQ_API_KEY must be provided"
  if (!fmpApiKey) 
    throw "FMP_API_KEY must be provided"

  const llm = openAIApiKey
    ? new ChatOpenAI({ 
      openAIApiKey, 
      model: "gpt-4", 
      temperature: 0 
    }) 
    : new ChatGroq({
      apiKey: groqApiKey,
      temperature: 0,
      model: "llama3-70b-8192",
    })

  /**
   * @note this getMockTranscript function exists because FMP's transcript endpoint is behind a pay wall :(
   */
  const getMockTranscript = _.memoize(async ({
    ticker,
    year,
    quarter,
  }: { ticker: string, quarter?: number, year?: number }) => {
    const { content } = await PromptTemplate
      .fromTemplate(`Generate a fake transcript for {ticker}'s conference call {period}`)
      .pipe(llm)
      .invoke({ ticker, period: year && quarter ? `for Q${quarter} ${year}` : "" })
    return content as string
  }, ({ ticker, year, quarter }) => `${ticker}-${year}-${quarter}`)

  const earningsCallKnowledgeRetriever = new DynamicStructuredTool({
    name: "earnings_call_knowledge_retriever",
    description: `
      Retrieves relevant information about the company's earnings, business and operations in a given quarter from the company's earnings call for the period.
      Information may include sales of particular products, segments, company initiatives, market conditions, etc...
      Information that is not available in financial statements can often be found here, so this can also be used if no other information is available or could not be retrieved.
    `,
    schema: z.object({
      originalPrompt: z.string().describe("The question from the user which you are trying to answer."),
      ticker: z.string().describe("The company ticker to retrieve earnings call transcript for."),
      year: z.number().optional(),
      quarter: z.number().optional(),
    }),
    callbacks: [new ConsoleCallbackHandler()],
    func: async (inputs) => {
      const { originalPrompt, ticker, year, quarter } = inputs
      const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings())
      const rawTranscript = await getMockTranscript({ ticker, year, quarter })
      if (!rawTranscript) return `Could not retrieve any transcript for ${ticker}`
      const documents = await new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        separators: ['\n\n', "\n", ' ', ''],
        chunkOverlap: 50,
      }).createDocuments([rawTranscript as string])
      await vectorStore.addDocuments(documents.map(doc => ({
        ...doc, 
        pageContent: `The following was found in ${ticker}'s conference call: \n...${doc.pageContent}` 
      })))
      const similarDocs = await vectorStore.similaritySearch(originalPrompt, 10)
      const { content } = await PromptTemplate
        .fromTemplate(`Summarize the following text in 200 words. \n The text: {snippets}`)
        .pipe(llm)
        .invoke({ snippets: similarDocs.map(d => d.pageContent).join("\n") })
      return content as string
    }
  })

  const financialStatementRetriever = new DynamicStructuredTool({
    name: "financial_statement_retriever",
    description: "Retrieves the relevant fincial statement data.",
    schema: z.object({
      statementType: z
        .enum(["balance-sheet-statement", "income-statement", "cash-flow-statement"])
        .describe(`
          The type of financial statement to retrieve.
          - income-statement: Income statments show the company's revenues and expenses during a particular period. It indicates how the revenues are transformed into the net income or net profit.
          - balance-sheet-statement: A balance sheet is a summary of the financial balances of the company, including assets, liabilities and shareholders' equity.
          - cash-flow-statement: A cash flow statement summarizes the amount of cash and cash equivalents entering and leaving a company, including cashflow from operations, financing and investing activities.
        `),
      ticker: z.string().describe("The stock ticker to perform the operation on."),
      period: z.enum(["annual", "quarter"]).describe("The period of the statements.").optional(),
      limit: z.number().describe("Number of statements to request.").optional().default(1)
    }),
    callbacks: [new ConsoleCallbackHandler()],
    func: async (args) => {
      const { statementType, ticker, limit, period } = args
      const url = new URL(`https://financialmodelingprep.com/api/v3/${statementType}/${ticker}`)
      url.searchParams.append("apikey", fmpApiKey)
      if (!!limit) url.searchParams.append("limit", encodeURI(limit.toString()))
      if (!!period) url.searchParams.append("period", encodeURI(period))
      const result = await fetch(url).then(res => res.json()).catch(() => null)
      return JSON.stringify(result)
    },
  })

  const tickerSearch = new DynamicStructuredTool({
    name: "ticker_search",
    description: "Finds ticker symbols and exchange information for equity securities and exchange-traded funds (ETFs) by searching with the company name.",
    schema: z.object({
      query: z.string().describe("Company name"),
      limit: z.number().describe("Maximum number of results to retrieve").optional(),
      exchange: z.enum(["NYSE", "NASDAQ"]).describe("The exchange to search").optional(),
    }),
    callbacks: [new ConsoleCallbackHandler()],
    func: async (args) => {
      const { exchange, query, limit } = args
      const url = new URL(`https://financialmodelingprep.com/api/v3/search-name`)
      url.searchParams.append("apikey", fmpApiKey)
      url.searchParams.append("query", query)
      if (!!exchange) url.searchParams.append("exchange", encodeURI(exchange))
      if (!!limit) url.searchParams.append("limit", encodeURI(limit?.toString()))
      const result = await fetch(url).then(res => res.json()).catch(() => null)
      return JSON.stringify(result)
    }
  })

  const tools = [financialStatementRetriever, tickerSearch, earningsCallKnowledgeRetriever]
  const agent = await createToolCallingAgent({
    llm,
    tools,
    prompt: ChatPromptTemplate.fromMessages([
      ["system", `
        It is ${new Date()} right now.
        You are a helpful assistant who assists users with answering questions about company stock financial performance and their business.
        If asked to perform any financial analysis, show the math. If you asked a tool, don't mention that you did so.
        If you could not retrieve any data from your tool, apologize and say nothing else.
        If you do not know the answer, or cannot find the relevant information, simply say so. DO NOT MAKE UP ANSWERS!
        Remember to respond professionally.
      `],
      ["placeholder", "{chat_history}"],
      ["human", "{input}"],
      ["placeholder", "{agent_scratchpad}"],
    ])
  })

  const chatHistory = new BufferMemory({
    memoryKey: "chat_history",
    returnMessages: true,
    outputKey: 'output',
    inputKey: 'input',
  })
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
    memory: new CombinedMemory({ memories: [chatHistory] }),
  })

  while (true) {
    const { Prompt: userPrompt } = await inquirer.prompt([{ name: "Prompt" }])
    try {
      const res = await agentExecutor.invoke({ input: userPrompt })
      console.log("==========================")
      console.log("[RESPONSE]:", res.output)
      console.log("==========================")
    } catch (err) {
      console.log("==========================")
      console.log("[RESPONSE]:", "Ooops, the power of your request is over 9000, I cannot handle it! Can you help guide me through the process step by step?")
      console.log("==========================")
    }
  }

})()