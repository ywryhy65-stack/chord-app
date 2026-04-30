import uvicorn

if __name__ == "__main__":
    # host="0.0.0.0" אומר לשרת להקשיב לכל מכשיר ברשת, לא רק למחשב עצמו
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)