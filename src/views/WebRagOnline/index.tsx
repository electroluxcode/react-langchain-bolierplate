import React, { useState } from 'react'
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
// import {  OpenAIEmbeddings } from "@langchain/openai";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { medicalData } from './medical';

export default function index() {

  const dataGet = async (inputValueTemp) => {
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
    for (let i of medicalStringHandle) {
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
    let embeddings = new OpenAIEmbeddings({
      openAIApiKey: "你的api key",
      configuration: {
        baseURL: "你的代理地址"
      },
      // modelName: "text-embedding-ada-002"
    })

    // 1.6.定义向量存储
    const vectorStore = await MemoryVectorStore.fromTexts(
      documents, [],
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
    const answer = await chain.invoke(
      inputValueTemp
    );
    setAnswer(answer);
  }
  const [quesion, setQuestion] = useState('提问区');
  const [answer, setAnswer] = useState('回答区');
  const [inputValue, setInputValue] = useState('');

  const handleSendMessage = () => {
    if (inputValue.trim() !== '') {
      setQuestion(inputValue)
      setInputValue('');
      dataGet(inputValue)
    }
  };


  return (
    <div className="chat-container bg-blue-1">
      <div className="chat-move">
        <div className="title">
          <h1 className='color-blue-1'>AI医疗助手</h1>
        </div>
        <div className="messages-container bg-white">
          <div
            className="message user"
          >
            {quesion}
          </div>
          <div
            className="message bot"
          >
            {answer}
          </div>
        </div>
        <div className="input-area">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
          />
          <button className='bg-blue-3 color-white' onClick={handleSendMessage}>发送</button>
        </div>
      </div>
    </div>
  );
}
