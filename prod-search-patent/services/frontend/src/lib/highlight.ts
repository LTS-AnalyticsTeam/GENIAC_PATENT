const escapeHtml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export const highlightText = (text: string, terms: string[]): string => {
  const uniqueTerms = Array.from(
    new Set(terms.filter((term) => term && term.trim().length > 1))
  );
  uniqueTerms.sort((a, b) => b.length - a.length);

  let highlighted = escapeHtml(text);
  for (const term of uniqueTerms) {
    const pattern = new RegExp(`(${escapeRegExp(term)})`, "gi");
    highlighted = highlighted.replace(pattern, "<mark>$1</mark>");
  }
  return highlighted;
};
