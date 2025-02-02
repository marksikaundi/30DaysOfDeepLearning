## MORE DETAILED EXPLAINATION

I'll break down FastAPI concepts in detail with examples and explanations.

### 1. Basic Setup and First Steps

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

**Explanation:**

- `FastAPI()` creates a new FastAPI application instance
- `@app.get("/")` is a decorator that tells FastAPI this function handles GET requests at the root URL "/"
- `async` makes the function asynchronous (FastAPI supports both sync and async)

### 2. Path Parameters (URL Parameters)

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id}

@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    return {"file_path": file_path}
```

**Explanation:**

- Path parameters are parts of the URL path enclosed in curly braces
- Type hints (`:int`, `:str`) provide automatic validation
- `:path` allows for full paths including slashes

### 3. Query Parameters

```python
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/items/")
async def read_items(
    skip: int = 0,                    # Required with default value
    limit: int = 10,                  # Required with default value
    q: str | None = None,             # Optional parameter
    price_gt: float | None = None     # Optional parameter for price filter
):
    items = [
        {"id": 1, "name": "Item 1", "price": 50},
        {"id": 2, "name": "Item 2", "price": 100},
    ]

    # Filter items based on query parameters
    if price_gt is not None:
        items = [item for item in items if item["price"] > price_gt]

    # Apply pagination
    return items[skip : skip + limit]
```

**Usage Examples:**

- `/items?skip=0&limit=10`
- `/items?price_gt=75`
- `/items?skip=0&limit=10&price_gt=75`

### 4. Request Body with Pydantic Models

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

# Define data models
class ItemBase(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True

app = FastAPI()

# Database simulation
items_db = []

@app.post("/items/", response_model=Item)
async def create_item(item: ItemCreate):
    # Create new item
    db_item = Item(
        **item.dict(),
        id=len(items_db) + 1,
        owner_id=1
    )
    items_db.append(db_item)
    return db_item
```

### 5. Complete CRUD Example with Error Handling

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Data Model
class Item(BaseModel):
    id: int | None = None
    name: str
    description: str | None = None
    price: float

# Database simulation
items_db = {}

# Create
@app.post("/items/", response_model=Item, status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    if item.id in items_db:
        raise HTTPException(
            status_code=400,
            detail="Item with this ID already exists"
        )
    items_db[item.id] = item
    return item

# Read
@app.get("/items/", response_model=List[Item])
async def read_items(skip: int = 0, limit: int = 10):
    return list(items_db.values())[skip : skip + limit]

@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    return items_db[item_id]

# Update
@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    items_db[item_id] = item
    return item

# Delete
@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(
            status_code=404,
            detail="Item not found"
        )
    del items_db[item_id]
    return {"message": "Item deleted successfully"}
```

### 6. Dependencies and Authentication

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional

app = FastAPI()

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Fake user database
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
    }
}

# Dependency for getting current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_users_db.get(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
```

### 7. Request Forms and File Uploads

```python
from fastapi import FastAPI, File, UploadFile, Form
from typing import List
import shutil
from pathlib import Path

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(
    file: UploadFile = File(...),
    description: str = Form(...)
):
    # Save uploaded file
    with Path(f"uploads/{file.filename}").open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "description": description,
        "content_type": file.content_type
    }

@app.post("/uploadfiles/")
async def create_upload_files(
    files: List[UploadFile] = File(...),
    note: str = Form(...)
):
    return {
        "filenames": [file.filename for file in files],
        "note": note
    }
```

### 8. Error Handling and Custom Responses

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

app = FastAPI()

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "type": "error"
        }
    )

# Custom error response
class CustomHTTPException(HTTPException):
    def __init__(self, status_code: int, detail: str, error_code: str):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 999:
        raise CustomHTTPException(
            status_code=404,
            detail="Item not found",
            error_code="ITEM_NOT_FOUND"
        )
    return {"item_id": item_id}
```

To run any of these examples:

1. Save the code in a file (e.g., `main.py`)
2. Install required packages:

```bash
pip install fastapi uvicorn
```

3. Run the server:

```bash
uvicorn main:app --reload
```

4. Access the API documentation:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

FastAPI automatically generates interactive API documentation based on your code and type hints, which is one of its most powerful features!
