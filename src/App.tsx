import { useRoutes} from 'react-router-dom'
import { AppSwrapper } from './style'

import routes from './router'
// import {Provider} from 'react-redux'


interface Function{
  getName:any
}

function App() {
  const element = useRoutes(routes)
  return (
    <>
      <AppSwrapper>{element}</AppSwrapper>
    </>
  )
}

export default App