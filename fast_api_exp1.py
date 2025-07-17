from fastapi import FastAPI

app=FastAPI(description='First Fast API class initialized')
# print(app.description)

@app.get('/greet')
def greet(name:str='Guest'):
    return {'message':f'Hello ! {name}'}
