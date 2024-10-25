import React from 'react'
import styles from "./index.module.less"
import { userCheck } from '@/api/docx'

export default function index() {
  userCheck().then(res => {
    console.log(res)
  })
  return (
    <div className={
      styles.container
    }>index</div>
  )
}
