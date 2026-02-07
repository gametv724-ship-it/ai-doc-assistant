import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
from rag_engine import DocumentRAG

st.set_page_config(page_title="📄 AI Ассистент", page_icon="📄", layout="wide")

st.title("📄 AI Ассистент для документов")
st.caption("Загружай PDF / Word / Excel → задавай вопросы → получай ответы")

with st.sidebar:
    st.header("⚙️ Настройки")
    
    use_docker = st.checkbox("Использовать Docker Qdrant", value=False, 
                            help="Если Docker не установлен, снимите галку")
    
    use_llm = st.checkbox("Использовать LLM (Groq)", value=False,
                         help="Нужен бесплатный API ключ с groq.com")
    
    groq_key = None
    if use_llm:
        groq_key = st.text_input("Groq API Key", type="password",
                                help="Получи на groq.com (бесплатно)")
    
    st.divider()
    
    # Кнопка очистки базы
    if st.button("🗑️ Очистить базу данных"):
        st.session_state.clear()
        st.rerun()
    
    st.caption("💡 Без LLM вернётся только найденный текст")

# Инициализация RAG
if 'rag' not in st.session_state:
    with st.spinner("Загружаю модели..."):
        st.session_state.rag = DocumentRAG(use_docker=use_docker)
        st.session_state.messages = []
        st.session_state.processed_files = set()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "📎 Загрузи документ", 
        type=['pdf', 'docx', 'xlsx', 'xls'],
        help="Поддерживаются PDF, Word, Excel"
    )

with col2:
    if uploaded_file:
        st.info(f"📄 **{uploaded_file.name}**\n\n{uploaded_file.size / 1024:.1f} KB")

if uploaded_file:
    temp_path = os.path.join("documents", uploaded_file.name)
    os.makedirs("documents", exist_ok=True)
    
    # Проверяем, был ли этот файл уже обработан
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if file_id not in st.session_state.processed_files:
        # Сохраняем файл
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Обрабатываем
        with st.spinner("🔄 Обрабатываю документ..."):
            result = st.session_state.rag.add_document(temp_path)
            st.success(result)
            # Помечаем как обработанный
            st.session_state.processed_files.add(file_id)
    else:
        st.info(f"✅ Документ уже загружен: {uploaded_file.name}")

st.divider()

st.subheader("💬 Задай вопрос по документу")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Источники"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Фрагмент {i}** • Релевантность: {src['score']:.2%} • {src['source']}")
                    st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])

if question := st.chat_input("Например: О чём этот документ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("🤔 Ищу ответ..."):
            response = st.session_state.rag.answer_question(
                question, 
                use_llm=use_llm, 
                groq_api_key=groq_key if use_llm else None
            )
            
            st.write(response["answer"])
            
            if response["sources"]:
                with st.expander("📚 Источники"):
                    for i, src in enumerate(response["sources"], 1):
                        st.caption(f"**Фрагмент {i}** • Релевантность: {src['score']:.2%} • {src['source']}")
                        st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response["answer"],
        "sources": response["sources"]
    })
