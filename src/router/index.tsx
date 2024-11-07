import React,{lazy, Suspense} from 'react';
import Transformer from '@/views/Transformer'
export interface IRouteBase {
    // 路由路径
    path: string;
    // 路由组件
    component?: any;
    // 302 跳转
    redirect?: string;
    // 路由信息
    meta: IRouteMeta;
    // 是否校验权限, false 为不校验, 不存在该属性或者为true 为校验, 子路由会继承父路由的 auth 属性
    auth?: boolean;
}
export interface IRouteMeta {
    title: string;
    icon?: string;
}

export interface IRoute extends IRouteBase {
    children?: IRoute[];
}



import Test from "@/views/Test"
import WebRagOnlineEasy from "@/views/WebRagOnlineEasy"
import WebRagOnline from "@/views/WebRagOnline"
const routes = [
  {
    path:"/",
    element:<Test></Test>
  },
  {
    path:"/WebRagOnline",
    element:<WebRagOnline></WebRagOnline>
  }, {
    path:"/WebRagOnlineEasy",
    element:<WebRagOnlineEasy></WebRagOnlineEasy>
  },
  {
    path: "/Transformer",
    element: <Transformer></Transformer>
  }
];
  
export default routes;
