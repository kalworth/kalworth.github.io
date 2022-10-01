## 羊了个羊通关方法汇总

### 更改请求关卡

这里使用抓包工具Fiddler

下载链接如下：[Fiddler | Web Debugging Proxy and Troubleshooting Solutions (telerik.com)](https://www.telerik.com/fiddler)

手机端可以使用HTTP catcher/Stream抓包,

正常安装即可

安装完成后首先我们需要对抓包工具进行配置

打开抓包工具的tools

![1663319169032](image/sheep/1663319169032.png)

选择内部的optionals选项，将下面勾选

再打开HTTPS选项

![1663319526942](image/sheep/1663319526942.png)

全部勾选并信任

点击actions

![1663319734766](image/sheep/1663319734766.png)

选择信任该证书

设置完成后我们应该就能成功抓取到本地的一些网络活动了

重启Fiddler后打开《羊了个羊》微信小程序，点击开始游戏

在抓包文件获取的网络请求中寻找sheep有关的活动

![1663320015967](image/sheep/1663320015967.png)

最终我们锁定在这两个请求上

看看详情

![1663320055553](image/sheep/1663320055553.png)

还真如网传一样通过map_id来获取地图

通过观察，我们会发现map_id共有两类

![1663321073972](image/sheep/1663321073972.png)

也就是80001和90016

接下来我们来打开FiddlerScript，开始编写脚本

转到 `OnBeforeRequest`

*`static function OnBeforeRequest(oSession: Session)` 在请求发送之前执行的函数，所以修改`request`的`header`和`body`就在此函数中。*

我们解决问题的思路就是修改通过第一关时的请求，将请求第二关地图改为请求简单的第一关请求

在 static function OnBeforeRequest(oSession: Session) 内添加如下代码

```js
if (oSession.url=="cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=90016") {
            oSession.url = "cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=80001";
        }
```

修改成功后重新打开小程序，完成第一关后发现仍然是第一关的内容

成功通关

![1663321413004](image/sheep/1663321413004.png)

~~（bug羊hhhhhh）~~




### 更改通关次数

有了更改地图的思路，那有没有更简单粗暴的方法呢？

我们能不能直接跳转到完成页面

还真能

将跳转到地图二的代码直接替换到gameover页面，但需要注意的是需要提交完成游戏的时间，我们可以自己来设置时间


```js
if (oSession.url="cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=90015") {????
	oSession.url = "cat-match.easygame2021.com/sheep/v1/game/game_over?rank_score=1&rank_state=1&rank_time=0&rank_role=1&skin=1"????
}
```

`rank_score rank_state rank_time` 也就是我们需要提交的数据

### 更改道具数量

逆向解包原小程序源码后对道具数量进行修改，最后打包为小程序文件

修改后破解的小程序链接：https://www.123pan.com/s/awP9-HtQnA 对应15号

目前小程序开发团队依然在不断进行迭代更新，所以链接生效日期有限


### 常见问题

***如果配置完Fiddler后仍然无法正常抓取，可能是因为本地网络配置问题，试试切换网络或者重启电脑***

***需要注意的是微信3.6以上的版本Fiddler不支持抓取，需要回滚微信版本，并删除当前小程序缓存***
