// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <string>
#include "JSONBase.h"
#include <fstream>
#include <sstream>


std::string JSONBase::Serialize() const
{
	rapidjson::StringBuffer ss;
	rapidjson::Writer<rapidjson::StringBuffer> writer(ss);		// Can also use Writer for condensed formatting
	if (Serialize(&writer))
		return ss.GetString();
	return "";
}

bool JSONBase::Deserialize(const std::string& s)
{
	rapidjson::Document doc;
	if (!InitDocument(s, doc))
		return false;

	Deserialize(doc);

	return true;
}

bool JSONBase::DeserializeFromFile(const std::string& filePath)
{
	std::ifstream f(filePath);
	std::stringstream buffer;
	buffer << f.rdbuf();
	f.close();

	return Deserialize(buffer.str());
}

bool JSONBase::SerializeToFile(const std::string& filePath)
{
	std::ofstream f(filePath);
	std::string s = Serialize();
	f << s;
	f.flush();
	f.close();

	return true;
}

bool JSONBase::InitDocument(const std::string& s, rapidjson::Document& doc)
{
	if (s.empty())
		return false;

	std::string validJson(s);

	return !doc.Parse(validJson.c_str()).HasParseError() ? true : false;
}