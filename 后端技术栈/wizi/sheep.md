## ���˸���ͨ�ط�������

### ��������ؿ�

����ʹ��ץ������Fiddler

�����������£�[Fiddler | Web Debugging Proxy and Troubleshooting Solutions (telerik.com)](https://www.telerik.com/fiddler)

�ֻ��˿���ʹ��HTTP catcher/Streamץ��,

������װ����

��װ��ɺ�����������Ҫ��ץ�����߽�������

��ץ�����ߵ�tools

![1663319169032](image/sheep/1663319169032.png)

ѡ���ڲ���optionalsѡ������湴ѡ

�ٴ�HTTPSѡ��

![1663319526942](image/sheep/1663319526942.png)

ȫ����ѡ������

���actions

![1663319734766](image/sheep/1663319734766.png)

ѡ�����θ�֤��

������ɺ�����Ӧ�þ��ܳɹ�ץȡ�����ص�һЩ������

����Fiddler��򿪡����˸���΢��С���򣬵����ʼ��Ϸ

��ץ���ļ���ȡ������������Ѱ��sheep�йصĻ

![1663320015967](image/sheep/1663320015967.png)

��������������������������

��������

![1663320055553](image/sheep/1663320055553.png)

����������һ��ͨ��map_id����ȡ��ͼ

ͨ���۲죬���ǻᷢ��map_id��������

![1663321073972](image/sheep/1663321073972.png)

Ҳ����80001��90016

��������������FiddlerScript����ʼ��д�ű�

ת�� `OnBeforeRequest`

*`static function OnBeforeRequest(oSession: Session)` ��������֮ǰִ�еĺ����������޸�`request`��`header`��`body`���ڴ˺����С�*

���ǽ�������˼·�����޸�ͨ����һ��ʱ�����󣬽�����ڶ��ص�ͼ��Ϊ����򵥵ĵ�һ������

�� static function OnBeforeRequest(oSession: Session) ��������´���

```js
if (oSession.url=="cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=90016") {
            oSession.url = "cat-match.easygame2021.com/sheep/v1/game/map_info?map_id=80001";
        }
```

�޸ĳɹ������´�С������ɵ�һ�غ�����Ȼ�ǵ�һ�ص�����

�ɹ�ͨ��

![1663321413004](image/sheep/1663321413004.png)

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
