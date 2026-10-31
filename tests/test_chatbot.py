
import unittest

from src.chatbot import FAQChatbot, FALLBACK


class TestFAQChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load embedding model only once
        cls.bot = FAQChatbot()

    def test_known_questions(self):
        cases = [
            ("What is the minimum number of tasks I need?", 0),
            ("Is one completed project enough?", 0),
            ("Which platform should host my code?", 1),
            ("Where do I send my completed internship work?", 2),
            ("Do I have to announce my internship on LinkedIn?", 3),
            ("Is a project demonstration video necessary?", 4),
            ("Which NLP method can match FAQ questions?", 5),
            ("What neural network can compose piano music?", 6),
            ("How do I track detected objects in a video?", 7),
        ]

        for question, faq_id in cases:
            with self.subTest(question=question):
                expected = self.bot.faqs[faq_id]["answer"]

                self.assertEqual(
                    self.bot.reply(question),
                    expected
                )

    def test_unknown_questions(self):
        questions = [
            "What will the weather be tomorrow?",
            "How can I reset my bank password?",
            "Who won yesterday's football match?",
            "What is the capital of Brazil?",
        ]

        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(
                    self.bot.reply(question),
                    FALLBACK
                )

    def test_similar_intents_are_distinguished(self):
        cases = [
            (
                "Where do I upload my source code files?",
                1
            ),
            (
                "Which GitHub repository should contain my project?",
                1
            ),
            (
                "Where do I submit my completed internship task?",
                2
            ),
            (
                "Where can I find the official project submission form?",
                2
            ),
        ]

        for question, faq_id in cases:
            with self.subTest(question=question):
                expected = self.bot.faqs[faq_id]["answer"]

                self.assertEqual(
                    self.bot.reply(question),
                    expected
                )

    def test_empty_and_invalid_questions(self):
        invalid_inputs = [
            "",
            "   ",
            None,
            123,
        ]

        for question in invalid_inputs:
            with self.subTest(question=question):
                self.assertEqual(
                    self.bot.reply(question),
                    FALLBACK
                )


if __name__ == "__main__":
    unittest.main()
