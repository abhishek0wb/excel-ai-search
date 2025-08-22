import { useState } from "react";
import axios from "axios";

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");

  const handleSubmit = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/query", { query });
      setAnswer(res.data.answer);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>Excel AI Search</h1>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask about your Excel file..."
      />
      <button onClick={handleSubmit}>Search</button>
      <p>Answer: {answer}</p>
    </div>
  );
}

export default App;
