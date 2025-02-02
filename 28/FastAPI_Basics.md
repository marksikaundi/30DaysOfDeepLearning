Basics of FastAPI with examples, starting from simple concepts and gradually moving to more advanced ones.

### 1. Basic FastAPI Setup and First Route

```python
from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Basic route (endpoint)
@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run with: `uvicorn main:app --reload`

### 2. Path Parameters

```python
from fastapi import FastAPI

app = FastAPI()

# Path parameter
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# Path parameter with string
@app.get("/items/{item_name}")
async def read_item(item_name: str):
    return {"item_name": item_name}
```

### 3. Query Parameters

```python
from fastapi import FastAPI

app = FastAPI()

# Query parameters
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Optional query parameters
@app.get("/products/")
async def read_products(name: str | None = None, price: float | None = None):
    return {
        "name": name,
        "price": price
    }
```

### 4. Request Body with Pydantic Models

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Define data model
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

# POST endpoint with request body
@app.post("/items/")
async def create_item(item: Item):
    return item

# PUT endpoint with path parameter and request body
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}
```

### 5. Form Data and File Uploads

```python
from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()

# Handle form data
@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}

# Handle file uploads
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

# Handle multiple files
@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}
```

### 6. Headers and Cookies

```python
from fastapi import FastAPI, Header, Cookie

app = FastAPI()

# Read headers
@app.get("/headers/")
async def read_headers(user_agent: str | None = Header(default=None)):
    return {"User-Agent": user_agent}

# Read cookies
@app.get("/cookies/")
async def read_cookies(session: str | None = Cookie(default=None)):
    return {"session": session}
```

### 7. Error Handling

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"foo": "The Foo Item"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
            headers={"X-Error": "Item not found"},
        )
    return {"item": items[item_id]}
```

### 8. Dependencies

```python
from fastapi import FastAPI, Depends, HTTPException

app = FastAPI()

async def verify_token(x_token: str = Header()):
    if x_token != "fake-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")
    return x_token

async def verify_key(x_key: str = Header()):
    if x_key != "fake-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key

@app.get("/items/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

### 9. Complete Example with Multiple Features

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Sample API")

# Data Model
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Database simulation
items_db = {}

# Dependencies
async def verify_admin(admin_token: str = Header(None)):
    if admin_token != "admin-secret":
        raise HTTPException(status_code=403, detail="Not authorized")
    return True

# CRUD Operations
@app.post("/items/", response_model=Item)
async def create_item(item: Item):
    items_db[item.name] = item
    return item

@app.get("/items/", response_model=List[Item])
async def read_items(skip: int = 0, limit: int = 10):
    return list(items_db.values())[skip : skip + limit]

@app.get("/items/{item_name}", response_model=Item)
async def read_item(item_name: str):
    if item_name not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_name]

@app.put("/items/{item_name}", response_model=Item)
async def update_item(item_name: str, item: Item):
    if item_name not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_name] = item
    return item

@app.delete("/items/{item_name}")
async def delete_item(
    item_name: str, is_admin: bool = Depends(verify_admin)
):
    if item_name not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_name]
    return {"message": "Item deleted"}
```

To test these endpoints, you can:

1. Run the server: `uvicorn main:app --reload`
2. Access the automatic interactive API documentation at `http://localhost:8000/docs`
3. Use the Swagger UI to test your endpoints
4. Use tools like curl or Postman to make requests

FastAPI provides many more features like:

- Background tasks
- WebSockets
- Security and authentication
- CORS (Cross-Origin Resource Sharing)
- Middleware
- Static files

This should give you a good foundation to start building APIs with FastAPI!
