---
title: sheep
tags: TeXt
---
## ���˸���ͨ�ط�������

### ��������ؿ�

����ʹ��ץ������Fiddler

�����������£�[Fiddler | Web Debugging Proxy and Troubleshooting Solutions (telerik.com)](https://www.telerik.com/fiddler)

�ֻ��˿���ʹ��HTTP catcher/Streamץ��,

������װ����

��װ��ɺ�����������Ҫ��ץ�����߽�������

��ץ�����ߵ�tools

![](http://forum.datawhale.club/uploads/default/original/2X/1/18fe9b9fc9796bee519d8e30c46643c017048c33.png)

ѡ���ڲ���optionalsѡ������湴ѡ

�ٴ�HTTPSѡ��

![](http://forum.datawhale.club/uploads/default/original/2X/d/d9b7ef382d6f5d3be046f601e3898ff7b44252c1.png)

ȫ����ѡ������

���actions

![](http://forum.datawhale.club/uploads/default/original/2X/6/64957cf1bb651ef8a6d12ed212d1ba2d77942b13.png)

ѡ�����θ�֤��

������ɺ�����Ӧ�þ��ܳɹ�ץȡ�����ص�һЩ������

����Fiddler��򿪡����˸���΢��С���򣬵����ʼ��Ϸ

��ץ���ļ���ȡ������������Ѱ��sheep�йصĻ

![](http://forum.datawhale.club/uploads/default/original/2X/a/a7686c973642d093cd73b2aa4c228c03b4a7fa31.png)

��������������������������

��������

![](http://forum.datawhale.club/uploads/default/original/2X/0/0ee0c596e274216ffa4767446efb310bc64b5f5f.png)

����������һ��ͨ��map_id����ȡ��ͼ

ͨ���۲죬���ǻᷢ��map_id��������

![](http://forum.datawhale.club/uploads/default/original/2X/b/bafca1baeebad8bc80521ecdb67ea229664f4609.png)

Ҳ����80001��90016

��������������FiddlerScript����ʼ��д�ű�

ת�� `OnBeforeRequest`

*`static function OnBeforeRequest(oSession: Session)` ��������֮ǰִ�еĺ����������޸� `request`�� `header`�� `body`���ڴ˺����С�*

���ǽ�������˼·�����޸�ͨ����һ��ʱ�����󣬽�����ڶ��ص�ͼ��Ϊ����򵥵ĵ�һ������

�� static function OnBeforeRequest(oSession: Session) ��������´���

```js
if (oSession.url=="cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=90016") {
            oSession.url = "cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=80001";
        }
```

�޸ĳɹ������´�С������ɵ�һ�غ�����Ȼ�ǵ�һ�ص�����

�ɹ�ͨ��

![](http://forum.datawhale.club/uploads/default/optimized/2X/5/55dfb605393362809acc1758d42a4d57a636b18b_2_421x750.png)

~~��bug��hhhhhh��~~

### ����ͨ�ش���

���˸��ĵ�ͼ��˼·������û�и��򵥴ֱ��ķ����أ�

�����ܲ���ֱ����ת�����ҳ��

������

����ת����ͼ���Ĵ���ֱ���滻��gameoverҳ�棬����Ҫע�������Ҫ�ύ�����Ϸ��ʱ�䣬���ǿ����Լ�������ʱ��

```js
if (oSession.url="cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=90015") {????
	oSession.url = "cat-match.easygame2021.com/sheep/v1/game/game_over?rank_score=1&rank_state=1&rank_time=0&rank_role=1&skin=1"????
}
```

`rank_score rank_state rank_time` Ҳ����������Ҫ�ύ������

### ���ĵ�������

������ԭС����Դ���Ե������������޸ģ������ΪС�����ļ�

�޸ĺ��ƽ��С�������ӣ�https://www.123pan.com/s/awP9-HtQnA ��Ӧ15��

ĿǰС���򿪷��Ŷ���Ȼ�ڲ��Ͻ��е������£�����������Ч��������

### ��������

***���������Fiddler����Ȼ�޷�����ץȡ����������Ϊ���������������⣬�����л����������������***

***��Ҫע�����΢��3.6���ϵİ汾Fiddler��֧��ץȡ����Ҫ�ع�΢�Ű汾����ɾ����ǰС���򻺴�***

```

```
