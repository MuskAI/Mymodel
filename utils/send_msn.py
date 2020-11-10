import urllib, urllib.request, sys
import ssl

def send_msn(epoch,f1):
    host = 'https://intlsms.market.alicloudapi.com'
    path = '/comms/sms/sendmsgall'
    method = 'POST'
    appcode = '096729f5b41b4830aeaa742e0d90d80b'
    querys = ''
    bodys = {}
    url = host + path

    bodys['callbackUrl'] = '''http://test.dev.esandcloud.com'''
    bodys['channel'] = '''0'''
    bodys['mobile'] = '''+8613329825566'''
    bodys['templateID'] = '''20201108001936'''
    epoch = str(epoch)
    f1 = str(f1)
    bodys['templateParamSet'] = [epoch,f1]
    post_data = urllib.parse.urlencode(bodys).encode("UTF8")
    request = urllib.request.Request(url, post_data)
    request.add_header('Authorization', 'APPCODE ' + appcode)
    # 根据API的要求，定义相对应的Content-Type
    request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    response = urllib.request.urlopen(request, context=ctx)
    content = response.read()
    if (content):
        print(content)

if __name__ == '__main__':
    send_msn(1,2)