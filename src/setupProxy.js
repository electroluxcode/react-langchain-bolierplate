let proxy = require('http-proxy-middleware')

module.exports = function (app)  {
  app.use(
  
    proxy.createProxyMiddleware('/test11111111111111111', {
      target: 'https://test11111111111111111/',
      changeOrigin: true,
      pathRewrite: {
        '^/test11111111111111111/': ''
      }
    }),
  )
}
