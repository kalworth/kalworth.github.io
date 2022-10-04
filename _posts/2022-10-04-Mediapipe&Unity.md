## ����mediapipe�Ķ�̬��׽�Լ�Unityͬ��ģ�Ͷ���

### Mediapipe����Լ�������׽ʵ��

MediaPipe�ǹȸ迪Դ�Ķ��������ѧϰ��ܣ���������˺ܶ�������̬��������⡢��Ĥ�ȸ��ָ�����ģ���Լ�����ѧϰ�㷨��

MediaPipe �ĺ��Ŀ���� C++ ʵ�֣���Ҫ�������Packet��Stream��Calculator��Graph�Լ���ͼSubgraph�����ݰ�������������ݵ�λ��һ�����ݰ���������ĳһ�ض�ʱ��ڵ�����ݣ�����һ֡ͼ���һС����Ƶ�źţ����������ɰ�ʱ��˳���������еĶ�����ݰ���ɣ�һ����������ĳһ�ض�Timestampֻ��������һ�����ݰ��Ĵ��ڣ��������������ڶ�����㵥Ԫ���ɵ�ͼ��������MediaPipe ��ͼ������ġ������ݰ���Source Calculator���� Graph Input Stream����ͼֱ����Sink Calculator ���� Graph Output Stream�뿪��

![](https://aijishu.com/img/bVVzx)

�������Ŀʹ��

#### MediaPipe����������ʵ����ý�嶯����׽��

```python
import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture('hello.mp4')

detector = PoseDetector()
posList = []
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
        posList.append(lmString)

    #print(len(posList))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        with open("MotionFile.txt", 'w') as f:
            f.writelines(["%s\n" % item for item in posList])
```

����ͨ��opencv�������ñ�����ý��Դ

ͬʱ����һ��������׽��������ȡ�����Ӧ��33���ؽڵ�

ÿ���ؽڵ��������£�

[0��x��y��z]

��Ҫע����ǣ�����y����opencv�ĳ���λ����unity��ģ�ĳ���λ�ò�ͬ��opencv�����Ͻǿ�ʼ��unity�����½ǿ�ʼ��������Ƕ�y��������˴���

```python
y = img.shape[0] - lm[2]
```

�����ݴ���posList�����Ǿ͵õ���unity���Զ�Ӧ�����йؽڽڵ�

������ǽ����ݻ����Ա��صķ�ʽ���浽MotionFile�ļ��ڡ�

### ��Unityͨ��ʵ��

ͨ�ŷ��棬����ʹ��Unity�ṩ֧�ֵ�Udpservice

#### �޸Ķ�����׽���룬����socket����ָ�����ض˿���unity����˿ڷ�������

```python
import cv2
from cvzone.PoseModule import PoseDetector
import socket
cap = cv2.VideoCapture(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5056)   

detector = PoseDetector()
posList = []  

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[1]},{img.shape[0] - lm[2]},{lm[3]},'
   
        print(lmString)
        date = lmString
        sock.sendto(str.encode(str(date)), serverAddressPort)

    cv2.imshow("Image", img)
```

ָ������5056�˿ڣ���unity����˽ű���������

#### ��Unity������Udp����ű���д

����UDPReceive�ű�

```csharp
using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPReceive : MonoBehaviour
{

	Thread receiveThread;
	UdpClient client;
	public int port = 5054;
	public bool Recieving = true;
	public bool printToConsole = false;
	public string data;


	public void Start()
	{

		receiveThread = new Thread(
			new ThreadStart(ReceiveData));
		receiveThread.IsBackground = true;
		receiveThread.Start();
	}

	private void ReceiveData()
	{

		client = new UdpClient(port);
		while (true) {
			if (Recieving)
			{
					try
					{
						IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
						byte[] dataByte = client.Receive(ref anyIP);
						data = Encoding.UTF8.GetString(dataByte);

						if (printToConsole) { print(data); }
					}
					catch (Exception err)
					{
						print(err.ToString());
					}
			}
		}
	}
}

```

������ָ������mediapipeʶ���python���͵ĵ�ʵʱ���ݡ�

���庯�� `ReciveData`�ڴ����Ľ����߳� `receiveThread`�����С�
ͨ���� `IsBackground`����Ϊ `true`�� ���߳�����Ϊ��̨�̣߳� ��ǰ̨������������

���� `UdpClinet`�Ͷ˿� `port`��`IPAdress.Any`��ʾ������ַ�����п���IP�� 0��ʾ���п��ö˿ڡ�
���� `receive`�������ݣ� ʹ�� `Encoding.UTF8.GetSting`ת��Ϊ�ַ�����

�������� `Recieving`�����Ƿ�������ݡ�

�������� `printToConsole`�����Ƿ����debug�����

��������˻����Ķ������������

### Unity�ڲ�����ʵ��

#### �����ؽڵ���Ӧ

����Actions�ű�

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Actions : MonoBehaviour
{
    // Start is called before the first frame update
    public UPD udpReceive;
    public GameObject[] bodyPoints;
    void Start()
    {
    
    }

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;
    
        print(data);
        string[] points = data.Split(',');
        print(points[0]);


        for (int i = 0; i < 32; i++)
        {

            float help;
            float.TryParse(points[0 + (i * 3)],out help);
            float x = help/100;
            float.TryParse(points[1 + (i * 3)],out help);
            float y = help/100;
            float.TryParse(points[2 + (i * 3)],out help);
            float z = help/300;

            bodyPoints[i].transform.localPosition = new Vector3(x, y, z);

        }

    }
}


```

���ܵ���̨���ݺ�Ϊ�����ű���д���ݣ����ڶ�����������������Ҫ��z�������⴦������ֵ�����3

#### ��д������Ⱦ�ű�

ͨ��Unity���õ�����Ⱦ������д���ӹؽڵĹ���

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LineCode : MonoBehaviour
{

    LineRenderer lineRenderer;

    public Transform origin;
    public Transform destination;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.1f;
    }
// ����������
    void Update()
    {
        lineRenderer.SetPosition(0, origin.position);
        lineRenderer.SetPosition(1, destination.position);
    }
}

```

ͬʱ���������Ⱦ����������MediaPipe�Ĳɼ���������Ӧ�Ĺ�����󶨡�

����ʵ�����

> �������� ��Pache([https://pache-ak.github.io/pache.github.io/](https://pache-ak.github.io/pache.github.io/)) Azula([https://limafang.github.io/Azula_blogs.github.io/](https://limafang.github.io/Azula_blogs.github.io/))
>
> �ο�ѧϰ���ݣ�https://www.youtube.com/watch?v=BtMs0ysTdkM
>
