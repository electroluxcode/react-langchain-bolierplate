import React from 'react'
import { useState} from 'react'

import { pipeline, env } from '@xenova/transformers';
env.useCustomCache = false
env.useFSCache = false
env.localModelPath = "/"
export default function index() {

  const dataGet = async (inputValueTemp) => {
    
    setAnswer("加载中");
    const translator = await pipeline('translation', 'nllb-200-distilled-600M');
    const output = await (translator as any)(inputValueTemp, {
      src_lang: 'zho_Hans', // Chinese
      tgt_lang: 'eng_Latn', // English
    }) ;
    setAnswer(output[0].translation_text);
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
    <div className="chat-container bg-black-1">
      <div className="chat-move">
        <div className="title">
          <h1 className='color-white-1'>AI翻译助手</h1>
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
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
          />
          <button className='bg-blue-3 color-white' onClick={handleSendMessage}>发送</button>
        </div>
      </div>
    </div>
  );
}
