import React, { useState } from 'react'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { useEffect } from 'react';
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { medicalData } from './medical';
export default function index() {
  const dataGet = async () => {
    // 1.2.数据预处理
    const medicalStringHandle = medicalData.map((e) => {
      return `病名是${e.name}, 不要吃${e.not_eat}, 应该要检查 ${e.check} ,用药一般推荐${e.drug_detail.join(",")}`
    })
    // 1.3.分词器
    const splitter = RecursiveCharacterTextSplitter.fromLanguage("html", {
      // 切分的最大长度
      chunkSize: 1000,
      // 相邻两个chunk之间的重叠token数量
      chunkOverlap: 20
    })
    let documents = []
    for(let i of medicalStringHandle) {
      const tempDoc = await splitter.splitText(i)
      documents = [...documents, ...tempDoc]
    }
    
    // 1.4.定义模型
    const model = new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: "sk-mv1Cw02wD7WgJPXD0IboyowxxT0XZ9jNt1pziHnQOW6A3XWj",
      configuration: {
        baseURL: "https://api.openai-proxy.org/v1"
      }
    });

    // 1.5.定义 embedding
    let embeddings =  new OpenAIEmbeddings({
      openAIApiKey: "你的api key",
      configuration: {
        baseURL: "你的代理地址"
      },
      // modelName: "text-embedding-ada-002"
    })

    // 1.6.定义向量存储
    const vectorStore = await MemoryVectorStore.fromTexts(
      documents,[],
      embeddings
    );

    const vectorStoreRetriever = vectorStore.asRetriever();

    // 1.7.定义prompt
    const SYSTEM_TEMPLATE = `使用上下文信息来回答最后的问题。如果你不知道答案，就直接说“我不知道”，不要试图编造答案。
    ----------------
    {context}`;

    const prompt = ChatPromptTemplate.fromMessages([
      ["system", SYSTEM_TEMPLATE],
      ["human", "{question}"],
    ]);

    // 1.8.构造问答链
    const chain = RunnableSequence.from([
      {
        context: vectorStoreRetriever,
        question: new RunnablePassthrough(),
      },
      prompt,
      model,
      new StringOutputParser(),
    ]);
    let quesion =  "肺转移瘤要怎么用药"
    const answer = await chain.invoke(
      quesion
    );

    console.log({ quesion,answer,documents,embeddings });
  }
  useEffect(() => {
    dataGet()
  })

  return (
    <div>打开控制台查看输出</div>
  );
}
