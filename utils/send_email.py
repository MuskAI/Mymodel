# smtplib 用于邮件的发信动作
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# email 用于构建邮件内容
from email.header import Header

import time

class SendMail:
    def __init__(self,header, context):
        # 用于构建邮件头

        # 发信方的信息：发信邮箱，QQ 邮箱授权码
        self.from_addr = 'haoranchr@163.com'
        self.password = 'EKYBQZDKANDPISHR'

        # 收信方邮箱
        self.to_addr = '1553442305@qq.com'

        # 发信服务器
        self.smtp_server = 'smtp.163.com'
        self.send_msg(header=header, context=context)
    def send_msg(self, header, context):
        # 邮箱正文内容，第一个参数为内容，第二个参数为格式(plain 为纯文本)，第三个参数为编码
        context = str(context)
        msg = MIMEText(context, 'plain',
                       'utf-8')

        # 邮件头信息
        msg['From'] = Header(self.from_addr)
        msg['To'] = Header(self.to_addr)
        msg['Subject'] = Header(header_name=header)

        # 开启发信服务，这里使用的是加密传输
        server = smtplib.SMTP_SSL(self.smtp_server)
        server.connect(self.smtp_server, 465)
        # 登录发信邮箱
        server.login(self.from_addr, self.password)
        # 发送邮件
        server.sendmail(self.from_addr, self.to_addr, msg.as_string())
        # 关闭服务器
        server.quit()

if __name__ == '__main__':
    now_time = time.asctime(time.localtime(time.time()))
    topic = 'Topic:'
    pages = ''
    context = '%s\n%s\n%s\n' %(now_time,topic,pages)
    SendMail(context)
